# GBSG2 Dataset

This directory contains [`gbsg2_survival.csv`](gbsg2_survival.csv), a numerically encoded version of the publicly documented **German Breast Cancer Study Group 2 (GBSG2)** dataset. I use these data for my self-directed study and implementation of survival machine-learning methods in Python.

## Dataset overview

| Item | Description |
| --- | --- |
| Patients | 686 women with breast cancer |
| Predictors | 8 baseline and treatment variables |
| Outcome | Recurrence-free survival (RFS) |
| Observed events | 299 (43.6%) |
| Right-censored observations | 387 (56.4%) |
| Observed follow-up | 8–2,659 days |
| Missing values | None in this CSV |

## Variable dictionary

| Column | Description | Units or coding |
| --- | --- | --- |
| `id` | Patient record identifier (not used as a predictor) | Integers 1–686 |
| `horTh` | Hormonal therapy | `0` = no; `1` = yes |
| `age` | Age | Years |
| `menostat` | Menopausal status | `1` = premenopausal; `2` = postmenopausal |
| `tsize` | Tumor size | mm |
| `tgrade` | Tumor grade | `1` = grade I; `2` = grade II; `3` = grade III |
| `pnodes` | Number of positive lymph nodes | Count |
| `progrec` | Progesterone receptor measurement | fmol |
| `estrec` | Estrogen receptor measurement | fmol |
| `time` | Observed time to the RFS endpoint or censoring | Days |
| `cens` | Event indicator | `1` = observed event; `0` = right-censored |

**Event coding:** Despite its name, `cens` uses `1` for an observed event and `0` for right censoring. The analysis follows this encoding throughout.

The categorical variables are stored as numeric codes in this CSV. Other published GBSG2 versions may use factor labels; see the [GBSG dataset documentation](https://search.r-project.org/CRAN/refmans/mfp/html/GBSG.html) for the corresponding codes.

## Use in this repository

The analysis in [`main.py`](../main.py) reads the data with `id` as the row index:

```python
import pandas as pd

data = pd.read_csv("data/gbsg2_survival.csv", encoding="utf-8-sig", index_col=0)
```

The eight predictors form the model inputs, while `time` and `cens` define the survival outcome. The workflow creates training, validation, and held-out test sets. Continuous predictors are standardized using parameters fitted on the training set only. No missing-value imputation is needed for this CSV.

## Data sources and references

Dataset documentation:

- [TH.data — GBSG2](https://search.r-project.org/CRAN/refmans/TH.data/html/GBSG2.html): dataset background and variable descriptions.
- [mfp — GBSG](https://search.r-project.org/CRAN/refmans/mfp/html/GBSG.html): numeric variable coding.
- [scikit-survival — `load_gbsg2`](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.datasets.load_gbsg2.html): survival-analysis dataset documentation.

Relevant publications:

1. Schumacher M, Bastert G, Bojar H, et al. Randomized 2 × 2 trial evaluating hormonal treatment and the duration of chemotherapy in node-positive breast cancer patients. *Journal of Clinical Oncology*. 1994;12(10):2086–2093. [doi:10.1200/JCO.1994.12.10.2086](https://doi.org/10.1200/JCO.1994.12.10.2086).
2. Sauerbrei W, Royston P. Building multivariable prognostic and diagnostic models: transformation of the predictors by using fractional polynomials. *Journal of the Royal Statistical Society: Series A*. 1999;162(1):71–94. [doi:10.1111/1467-985X.00122](https://doi.org/10.1111/1467-985X.00122).

**Attribution and reuse:** The clinical observations originate from third-party research. This repository does not claim ownership of the underlying data. When reusing the dataset, cite the original research and dataset documentation. The license for this repository's code does not automatically apply to the underlying dataset; consult the terms of the relevant upstream distribution.
