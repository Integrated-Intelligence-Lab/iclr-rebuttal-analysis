"""Streamlined no-pos_before ablation. Uses load_feature_matrix from
rebuttal_rf_prediction.py, fits LR + RF on temporal split (train 2024 -> test 2025)
plus GroupKFold(5) on combined years, reports AUC + AUPRC + log_loss + Brier
with bootstrap 95% CIs. Skips permutation-importance (separate question).
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src"))

print("importing sklearn...", flush=True)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
)
print("sklearn ok", flush=True)

print("importing rebuttal_rf_prediction.load_feature_matrix...", flush=True)
from analyze.rebuttal_rf_prediction import load_feature_matrix
print("ok", flush=True)


RNG_SEED = 50


def fit_rf(X, y):
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=20,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RNG_SEED,
    )
    rf.fit(X, y)
    return rf


def fit_lr(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG_SEED)
    lr.fit(Xs, y)
    return lr, sc


def proba(model, X, scaler=None):
    if scaler is not None:
        return model.predict_proba(scaler.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]


def boot_ci(y, score, metric, n=1000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    pt = float(metric(y, score))
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        try:
            boots.append(metric(y[idx], score[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pt, float(lo), float(hi)


def evaluate(df_train, df_test, feats, label):
    X_tr, y_tr = df_train[feats].values, df_train["score_increase"].to_numpy()
    X_te, y_te = df_test[feats].values, df_test["score_increase"].to_numpy()
    out = {"horizon": label, "n_features": len(feats),
           "n_train": len(X_tr), "n_test": len(X_te),
           "base_rate_test": float(y_te.mean())}
    for name, fitter in [("rf", "rf"), ("logistic", "lr")]:
        if fitter == "rf":
            m = fit_rf(X_tr, y_tr); s = proba(m, X_te)
        else:
            m, sc = fit_lr(X_tr, y_tr); s = proba(m, X_te, sc)
        auc = boot_ci(y_te, s, roc_auc_score)
        ap = boot_ci(y_te, s, average_precision_score)
        ll = boot_ci(y_te, s, log_loss)
        br = boot_ci(y_te, s, brier_score_loss)
        out[name] = {
            "auc_roc": auc[0], "auc_roc_ci": (auc[1], auc[2]),
            "auprc": ap[0],   "auprc_ci": (ap[1], ap[2]),
            "log_loss": ll[0], "log_loss_ci": (ll[1], ll[2]),
            "brier": br[0],    "brier_ci": (br[1], br[2]),
        }
    return out


def cv_eval(df, feats):
    X = df[feats].values
    y = df["score_increase"].to_numpy()
    g = df["paper_key"].to_numpy()
    metrics = {"rf": [], "logistic": []}
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        for name, fitter in [("rf", "rf"), ("logistic", "lr")]:
            if fitter == "rf":
                m = fit_rf(X[tr], y[tr]); s = proba(m, X[te])
            else:
                m, sc = fit_lr(X[tr], y[tr]); s = proba(m, X[te], sc)
            metrics[name].append({
                "auc_roc": roc_auc_score(y[te], s),
                "auprc": average_precision_score(y[te], s),
                "log_loss": log_loss(y[te], s),
                "brier": brier_score_loss(y[te], s),
            })
    summary = {}
    for name, fold_metrics in metrics.items():
        d = {}
        for k in ["auc_roc", "auprc", "log_loss", "brier"]:
            vals = [m[k] for m in fold_metrics]
            d[f"{k}_mean"] = float(np.mean(vals))
            d[f"{k}_std"] = float(np.std(vals))
        summary[name] = d
    return summary


def main():
    horizons = ["pre", "pre_no_score"]
    summary = []
    for h in horizons:
        print(f"\n{'='*60}\n  Horizon: {h}\n{'='*60}", flush=True)
        df_2024 = load_feature_matrix(2024, h)
        df_2025 = load_feature_matrix(2025, h)
        feats = [c for c in df_2024.columns if c not in ("paper_key", "score_increase")]
        print(f"features ({len(feats)}): {feats}", flush=True)
        print(f"n_2024={len(df_2024)} pos={int(df_2024['score_increase'].sum())}, "
              f"n_2025={len(df_2025)} pos={int(df_2025['score_increase'].sum())}", flush=True)

        print("temporal split (train 2024 -> test 2025)...", flush=True)
        t = evaluate(df_2024, df_2025, feats, h)
        print(json.dumps(t, indent=2, default=str), flush=True)

        print("GroupKFold(5) on combined...", flush=True)
        df_24g = df_2024.assign(paper_key="2024_" + df_2024["paper_key"].astype(str))
        df_25g = df_2025.assign(paper_key="2025_" + df_2025["paper_key"].astype(str))
        cv = cv_eval(pd.concat([df_24g, df_25g], ignore_index=True), feats)
        print(json.dumps(cv, indent=2), flush=True)

        for model in ("rf", "logistic"):
            row = {
                "horizon": h, "model": model, "n_features": len(feats),
                "n_train": len(df_2024), "n_test": len(df_2025),
            }
            row.update({k: t[model][k] for k in ("auc_roc", "auc_roc_ci", "auprc", "auprc_ci",
                                                  "log_loss", "log_loss_ci", "brier", "brier_ci")})
            row.update({f"cv_{k}": v for k, v in cv[model].items()})
            summary.append(row)

    pd.DataFrame(summary).to_csv(
        repo / "data" / "results" / "rebuttal_rf_prediction" / "tables"
        / "incremental_value_summary_no_score_ablation.csv", index=False,
    )
    print("\nSaved: data/results/rebuttal_rf_prediction/tables/incremental_value_summary_no_score_ablation.csv", flush=True)


if __name__ == "__main__":
    main()
