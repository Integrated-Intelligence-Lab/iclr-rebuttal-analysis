"""
Convert the per-year parquet review tables in data/processed/parquet/ to CSV.

Use this when you want to inspect the data in Excel, awk, or another tool
that does not read parquet. The analysis scripts in this repo all read the
parquet files directly, so converting is optional.

Usage
-----
    python scripts/parquet_to_csv.py            # convert all years
    python scripts/parquet_to_csv.py --year 2024
    python scripts/parquet_to_csv.py --out-dir /tmp/iclr_csv

Trade-off note: CSV is roughly 4x larger than parquet on this dataset and
loses dtype information. Re-running an analysis on the round-tripped CSV
can produce small floating-point drift relative to the parquet original.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import pandas as pd

from core.paths import PROCESSED_DATA_DIR

PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
DEFAULT_YEARS = [2023, 2024, 2025, 2026]


def convert(year: int, out_dir: Path) -> Path:
    src = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Missing parquet file: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"iclr_{year}_reviews.csv"
    df = pd.read_parquet(src)
    df.to_csv(dst, index=False)
    print(f"  {src.name}  ({src.stat().st_size / 1e6:.1f} MB)"
          f"  ->  {dst.name}  ({dst.stat().st_size / 1e6:.1f} MB)"
          f"  [{len(df):,} rows x {df.shape[1]} cols]")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--year", type=int, choices=DEFAULT_YEARS,
                    help="Convert a single year (default: all four)")
    ap.add_argument("--out-dir", type=Path, default=PARQUET_DIR,
                    help="Output directory (default: data/processed/parquet/)")
    args = ap.parse_args()

    years = [args.year] if args.year else DEFAULT_YEARS
    for y in years:
        convert(y, args.out_dir)


if __name__ == "__main__":
    main()
