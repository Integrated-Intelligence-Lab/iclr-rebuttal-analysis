# Test 2 expansion: n=952 -> full classified ICLR 2024 cohort
## Cohort sizes
- Original Test 2 (stratified): n = 952 (base rate 41.5% increase)
- Expanded Test 2 (full 2024 classified): n = 5,730 (base rate 45.8% increase)
- Net additional rows: 4,778

## Polarity preservation (vs codebook-declared polarity)
- Original Test 2: 43 of 44 factors match codebook polarity
- Expanded Test 2: 44 of 44
- Original mismatched factor IDs: ['F40']
- Expanded mismatched factor IDs: []

## Doubly-Bonferroni-significant (Test 1 perm_p < 0.0011 AND Test 2 perm_p < 0.0011)
- With original n=952 Test 2 (naive perm): 0 (0 positive, 0 negative)
- Expanded full-2024, naive (row-level) perm: 0 (0 positive, 0 negative)
- Expanded full-2024, paper-clustered (block) perm: 0 (0 positive, 0 negative)
- Newly doubly-Bonf under expanded NAIVE: []
- Lost from doubly-Bonf under expanded NAIVE: []
- Newly doubly-Bonf under expanded CLUSTERED: []
- Lost from doubly-Bonf under expanded CLUSTERED: []

## Per-factor changes (sorted by absolute change in log-OR)
| factor | pol | OR (952) | OR (full) | dlogOR | perm_p (952) | perm_p naive (full) | perm_p clust (full) | doubly orig | doubly new naive | doubly new clust |
|--------|-----|----------|-----------|--------|--------------|---------------------|---------------------|-------------|------------------|------------------|
| F40 | negative | 1.43 | 0.36 | -1.37 | 0.333 | 0.0005 | 0.0005 | no | no | no |
| F37 | negative | 0.25 | 0.09 | -0.98 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F34 | negative | 0.96 | 0.37 | -0.95 | 1 | 0.0005 | 0.0005 | no | no | no |
| F24 | negative | 0.14 | 0.36 | +0.93 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F17 | positive | 4.32 | 1.94 | -0.80 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F35 | negative | 0.08 | 0.04 | -0.67 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F01 | positive | 3.08 | 6.00 | +0.67 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F32 | negative | 0.85 | 0.52 | -0.49 | 0.833 | 0.0005 | 0.0005 | no | no | no |
| F31 | negative | 0.61 | 0.38 | -0.47 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F30 | negative | 0.86 | 0.56 | -0.44 | 0.833 | 0.0005 | 0.0005 | no | no | no |
| F43 | negative | 0.58 | 0.39 | -0.40 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F04 | positive | 5.19 | 7.71 | +0.39 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F28 | negative | 0.46 | 0.31 | -0.39 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F33 | negative | 0.46 | 0.31 | -0.38 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F13 | positive | 1.53 | 2.18 | +0.35 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F36 | negative | 0.36 | 0.25 | -0.34 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F26 | negative | 0.28 | 0.37 | +0.28 | 0.167 | 0.003 | 0.002 | no | no | no |
| F27 | negative | 0.49 | 0.38 | -0.26 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F29 | negative | 0.48 | 0.38 | -0.24 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F08 | positive | 2.56 | 2.02 | -0.24 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F15 | positive | 1.48 | 1.88 | +0.23 | 0.333 | 0.0005 | 0.0005 | no | no | no |
| F41 | negative | 0.35 | 0.44 | +0.23 | 0.333 | 0.0895 | 0.0985 | no | no | no |
| F10 | positive | 1.78 | 2.24 | +0.23 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F05 | positive | 2.34 | 2.94 | +0.23 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F18 | positive | 1.67 | 2.09 | +0.22 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F21 | positive | 2.61 | 2.15 | -0.20 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F39 | negative | 0.37 | 0.30 | -0.20 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F14 | positive | 1.86 | 2.22 | +0.17 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F38 | negative | 0.26 | 0.31 | +0.17 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F02 | positive | 3.57 | 3.01 | -0.17 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F09 | positive | 3.36 | 3.97 | +0.17 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F22 | positive | 1.55 | 1.82 | +0.16 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F06 | positive | 1.87 | 2.18 | +0.16 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F25 | negative | 0.23 | 0.20 | -0.15 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F44 | negative | 0.78 | 0.67 | -0.15 | 0.667 | 0.0365 | 0.0295 | no | no | no |
| F03 | positive | 3.38 | 2.93 | -0.14 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F11 | positive | 1.40 | 1.52 | +0.09 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F07 | positive | 2.12 | 2.27 | +0.07 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F12 | positive | 1.11 | 1.19 | +0.06 | 0.5 | 0.0045 | 0.0045 | no | no | no |
| F20 | positive | 1.62 | 1.57 | -0.03 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F42 | negative | 0.27 | 0.28 | +0.03 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F19 | positive | 1.81 | 1.78 | -0.02 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F16 | positive | 2.06 | 2.10 | +0.02 | 0.167 | 0.0005 | 0.0005 | no | no | no |
| F23 | positive | 2.11 | 2.11 | +0.00 | 0.167 | 0.0005 | 0.0005 | no | no | no |
