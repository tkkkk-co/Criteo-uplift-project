# Incremental Conversion Optimization Under a Fixed Targeting Budget

An end-to-end applied data science case study on the Criteo Uplift benchmark.
The decision is whom to target when only 20% of eligible users can be assigned
advertising. The objective is incremental conversions across the eligible
population.

## Dataset

The [Criteo AI Lab benchmark](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
contains 13,979,592 public v2.1 records assembled from advertising incrementality
tests. The pipeline audits every record and draws a uniform sample of 1,000,000
rows for modeling. It uses the 12 anonymized features and the treatment
assignment, with conversion as the primary outcome. Actual exposure is
post-assignment and excluded from model inputs. See [NOTICE.md](NOTICE.md) for
attribution and data-derived report licensing.

## Methodology

A single read of the source CSV validates the schema, computes SHA-256 and the
full-data audit, and draws separate modeling and C2ST samples. Modeling rows are
split 60/20/20 by treatment and outcome with disjoint source row IDs. Four
S/T/X-learner candidates are trained on Train and scored alongside the
Response Propensity baseline, the treated outcome prediction from T-XGBoost.

Validation freezes the policy with the highest mean HT Gain across 200
stratified, paired bootstrap draws at the preset 20% budget. Independent Test
loads the frozen models and reports AUUC, Traditional Qini Score, three
budget-specific gains, and paired intervals for HT Gain. Stability uses only
Train and Validation, across seeds 17, 29, and 83, three training fractions,
and a feature-group holdout.

## Key findings

- **Response Propensity** won Validation. **S-Logistic** ranked first among
  the four uplift candidates.
- On 200,000 held-out Test rows, Response Propensity at 20% coverage yielded
  **117.23 incremental conversions per 100,000 eligible users**
  (95% CI **[64.09, 166.21]**). Its paired difference versus Random was
  **+92.61 [50.02, 132.15]**.
- S-Logistic's paired Test difference versus Response Propensity was
  **+6.10 [−1.77, 14.61]** per 100,000. The interval includes zero.
- The frozen policy's HT point estimates at 10%, 20%, 30%, and 100% coverage
  were **106.27**, **117.23**, **121.34**, and **123.07** per 100,000.
  The next ten percentage points after 20% added 4.12, compared with 10.95
  for the 10%–20% interval. These are descriptive comparisons.

## Repository structure

| Path | Contents |
| --- | --- |
| `run_pipeline.py` | Eight-stage experiment entry point |
| `criteo_uplift/` | Data preparation, diagnostics, training, evaluation, stability, planning and reporting |
| `tests/` | Statistical and synthetic end-to-end acceptance tests |
| `reports/final/report.html` | Standalone research report |
| `data/raw/`, `artifacts/` | Git-ignored input and local experiment output |

## Reproducibility

Use Python 3.11. On macOS, XGBoost requires an OpenMP runtime, available via
`brew install libomp`. The tested direct dependency versions are listed in
[requirements-tested.txt](requirements-tested.txt). Put the uncompressed Criteo
v2.1 CSV at `data/raw/criteo-uplift-v2.1.csv`. The tests use synthetic data and
can run before obtaining the source file.

```bash
python -m pip install -e . -c requirements-tested.txt
python -m unittest discover -s tests -v

python run_pipeline.py \
  --data data/raw/criteo-uplift-v2.1.csv \
  --sample-size 1000000 --outcome conversion --seed 42 \
  --primary-metric gain_ht --primary-budget 0.20 --bootstrap 200 \
  --permutations 499 --c2st-sample 20000 --trees 150 \
  --output artifacts/final_conversion_1000000_seed42
```

Use an empty output directory for each run. The output contains the source
fingerprint, split audits, frozen selection, scores, aggregate estimates, and
the HTML report.

## Final project report

[Incremental Conversion Optimization Under a Fixed Targeting Budget](reports/final/report.html).
Download the HTML or open it from a checkout to read the Chinese-language
report offline. Figures and formulas are embedded.

## Limitations

The estimates describe treatment assignment in an anonymized, subsampled
public benchmark. The diagnostic evidence addresses observable imbalance.
The original experiment's full allocation design and user-level dependence
are unavailable. Intervals condition on fixed scores and the training
treatment proportion. Deployment economics and uplift performance require a
new randomized comparison in the target population.
