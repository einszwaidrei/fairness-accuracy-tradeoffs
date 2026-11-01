# Dataset Notes — ACSIncome (Illinois, 2018)

## Data source and prediction task

This study uses the ACSIncome task from [Folktables](https://github.com/socialfoundations/folktables), based on the 2018 American Community Survey (ACS) 1-Year Person Survey for Illinois. After the task's eligibility filters, the dataset contains **67,016 observations**.

The binary target indicates whether annual personal income exceeds $50,000:

- `0`: income ≤ $50,000
- `1`: income > $50,000

## Features and sensitive attribute

The models use nine input features: `AGEP`, `COW`, `SCHL`, `MAR`, `OCCP`, `POBP`, `RELP`, `WKHP`, and `RAC1P`.

- **Numerical:** `AGEP` (age) and `WKHP` (usual hours worked per week), standardized before model fitting.
- **Categorical:** the other seven features, encoded using one-hot encoding.

The ACS variable `SEX` is recorded as `1` (male) or `2` (female). It is excluded from the models' input features but retained separately to measure group disparities and apply fairness mitigation. In particular, group membership is used when computing reweighing weights, fitting and applying group-specific threshold rules, and enforcing the Exponentiated Gradient constraint during training.

## Dataset characteristics

| Characteristic | Share |
|---|---:|
| Male | 51.6% |
| Female | 48.4% |
| Income > $50,000 | 39.35% |
| Income > $50,000 among males | 47.74% |
| Income > $50,000 among females | 30.40% |

## Data splitting

The observations are split using random seed `42`, stratified by the target label:

| Split | Share | Observations | Purpose |
|---|---:|---:|---|
| Train | 60% | 40,209 | Fit baseline and mitigation models |
| Validation | 20% | 13,403 | Assess experiments and fit threshold post-processing |
| Test | 20% | 13,404 | Evaluate all configurations on held-out data |

The split is shared across experiments. Test labels are not used to fit models or select mitigation rules.

## Scope and limitations

- The sample includes only observations eligible for the ACSIncome task in Illinois in 2018; findings should not be generalized to all Illinois residents or the US population.
- `SEX` reflects two administrative response categories in these data and does not represent the full range of gender identities.
- Excluding `SEX` from predictors does not prevent other features from acting as proxies for group membership.
- Group differences in income or model outcomes do not, on their own, establish causal discrimination.
- Fairness results depend on the selected metrics, the two recorded groups, the study population, and the evaluation split.
