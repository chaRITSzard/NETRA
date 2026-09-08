# NETRA — Network Threat Recognition & Analysis

NETRA (नेत्र — "eye") is a multi-stage network intrusion detection system built to study how statistical analysis, machine learning, explainability, API deployment, and performance benchmarking can work together in a single security-ML pipeline.

The project uses the **NSL-KDD** dataset and follows a deliberate:

**R → Python → Julia**

architecture.

R is used for statistical feature analysis, Python handles feature engineering and machine-learning development, and Julia is used to benchmark a tree-based implementation for performance.

---

## Overview

NETRA classifies network traffic into five attack categories:

- **Normal**
- **DoS** — Denial of Service
- **Probe**
- **R2L** — Remote to Local
- **U2R** — User to Root

The system was designed around a practical ML workflow rather than treating model accuracy as the only objective.

It includes:

- Statistical feature analysis in R
- A reproducible Python preprocessing pipeline
- Random Forest and XGBoost baselines
- Optuna-based XGBoost tuning
- Stratified cross-validation
- Frozen evaluation on `KDDTest+`
- Failure and distribution-shift analysis
- SHAP explainability
- FastAPI inference
- Docker deployment
- API load testing
- Julia Random Forest benchmarking
- Reproducible artifacts and reports

---

## Architecture

```text
                         NSL-KDD
                            │
              ┌─────────────┴─────────────┐
              │                           │
         KDDTrain+                    KDDTest+
              │                           │
              ▼                           │
       ┌──────────────┐                   │
       │      R       │                   │
       │ Statistical  │                   │
       │   Analysis   │                   │
       └──────┬───────┘                   │
              │                           │
      feature_report.html                │
              │                           │
              ▼                           │
       feature_config.yaml               │
              │                           │
              ▼                           │
       ┌──────────────┐                   │
       │    Python    │                   │
       │              │                   │
       │ Preprocessing│                   │
       │      ↓       │                   │
       │  RF / XGBoost│                   │
       │      ↓       │                   │
       │   Optuna     │                   │
       │      ↓       │                   │
       │    SHAP      │                   │
       └──────┬───────┘                   │
              │                           │
              ├──────────────┐            │
              │              │            │
              ▼              ▼            │
         FastAPI          Julia           │
         + Docker       Benchmarking      │
              │              │             │
              ▼              ▼             │
          /predict       Random Forest     │
                             │             │
                             ▼             │
                       Performance Report  │
                                           │
                                           ▼
                                  Final KDDTest+
                                  Evaluation
```

The stages communicate through files rather than live cross-language calls. This keeps each stage independently reproducible and makes the handoff between statistical analysis, model development, and benchmarking explicit.

---

# Dataset

NETRA uses the **NSL-KDD** intrusion-detection dataset.

The dataset contains:

- **41 original traffic features**
- A traffic label
- A difficulty field

NETRA converts the original attack labels into five broader classes:

| Class | Description |
|---|---|
| Normal | Legitimate network traffic |
| DoS | Denial of Service |
| Probe | Reconnaissance / scanning |
| R2L | Remote to Local |
| U2R | User to Root |

Dataset files:

```text
data/raw/
├── KDDTrain+.txt
└── KDDTest+.txt
```

`KDDTest+` is reserved for final evaluation and is not used during model training or feature selection.

---

# 1. Statistical Analysis — R

The first stage uses R to investigate the dataset before machine-learning development.

The analysis includes:

- Data loading and cleaning
- Binary and five-class target construction
- Descriptive statistics
- Chi-square analysis
- ANOVA / Kruskal-Wallis testing
- Correlation analysis
- Redundancy detection
- Logistic-regression baseline
- Feature visualizations
- Feature recommendations

The resulting recommendations are exported as:

```text
r_analysis/reports/feature_report.html
```

and translated into:

```text
python_pipeline/config/feature_config.yaml
```

This creates an explicit handoff between statistical analysis and machine-learning preprocessing.

---

# 2. Python Preprocessing

The Python pipeline implements the feature policy derived from the R analysis.

The custom preprocessor is:

```python
NETRApreprocessor
```

The preprocessing strategy includes:

### Categorical features

Low-cardinality features:

```text
protocol_type
flag
```

are one-hot encoded.

The higher-cardinality:

```text
service
```

feature is frequency encoded.

### Numeric transformations

Logarithmic transformation is applied using:

```text
log1p(duration)
log1p(src_bytes)
log1p(dst_bytes)
```

Redundant features identified during the R analysis are removed.

The processed training data contains:

```text
125,973 rows
44 processed features
```

The preprocessor also ensures consistent feature ordering between training and inference.

---

# 3. Model Development

NETRA evaluates tree-based machine-learning models in Python.

Models investigated include:

- Random Forest
- XGBoost
- Weighted XGBoost
- Tuned weighted XGBoost

Five-fold stratified cross-validation was used during model development.

The tuned XGBoost model was optimized using **Optuna** with macro-F1 as the optimization objective.

The best Optuna configuration was:

```text
n_estimators       = 215
max_depth          = 10
learning_rate      = 0.0762595
subsample          = 0.96318
colsample_bytree   = 0.96496
weight_multiplier  = 1.58682
```

The best recorded Optuna macro-F1 was:

```text
0.955401
```

An independent five-fold evaluation of the selected configuration produced:

```text
Accuracy:    0.999087
Macro-F1:    0.945869
Weighted-F1: 0.999092
```

The difference between the Optuna objective and the independent CV result is retained in the project artifacts rather than silently reconciled.

---

# 4. Final Evaluation

The final evaluation uses the official:

```text
KDDTest+.txt
```

This dataset was kept separate from model development.

### Final five-class results

| Metric | Result |
|---|---:|
| Accuracy | **79.4846%** |
| Macro Precision | **0.813406** |
| Macro Recall | **0.582648** |
| Macro-F1 | **0.621465** |
| Weighted-F1 | **0.767775** |
| Binary ROC-AUC | **0.973011** |

### Per-class performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0.693965 | 0.972094 | 0.809814 |
| DoS | 0.967000 | 0.856300 | 0.908289 |
| Probe | 0.874518 | 0.656340 | 0.749882 |
| R2L | 0.989879 | 0.168504 | 0.287986 |
| U2R | 0.541667 | 0.260000 | 0.351351 |

The model performs strongly on Normal, DoS, and Probe traffic, but performance on R2L and U2R drops substantially.

The original project target of at least 95% five-class accuracy on `KDDTest+` was **not achieved**.

This result is intentionally reported rather than hidden.

---

# 5. Failure Analysis

The large gap between cross-validation performance and `KDDTest+` performance led to a dedicated failure-analysis stage.

The analysis identified substantial differences between training and test distributions.

Most notably, R2L and U2R classes exhibit strong distribution and subtype shifts.

Examples include:

- Different R2L subtype composition
- Test-only attack subtypes
- Changes in service distributions
- Changes in protocol distributions
- Significant shifts in traffic statistics

For R2L in particular, the training data is dominated by a small number of attack subtypes, while `KDDTest+` contains several subtypes that are absent from training.

This provides an important explanation for why near-perfect validation performance does not translate into comparable performance on the official test set.

The failure analysis is diagnostic only. It was not used to repeatedly optimize against `KDDTest+`.

---

# 6. Feature Ablation

Feature-ablation experiments were performed to determine whether restoring removed features improved generalization.

One candidate feature:

```text
dst_host_srv_count
```

appeared promising on an individual split.

However, five-fold cross-validation showed that adding it reduced macro-F1 and U2R performance.

Therefore the original feature policy was retained.

This prevents a single favorable split from driving feature-selection decisions.

---

# 7. Explainability — SHAP

NETRA uses SHAP to investigate the behavior of the tuned XGBoost model.

SHAP analysis was performed on a stratified sample of the training data.

The most influential processed features included:

| Rank | Feature |
|---:|---|
| 1 | `src_bytes` |
| 2 | `service` |
| 3 | `count` |
| 4 | `dst_bytes` |
| 5 | `flag_S0` |
| 6 | `dst_host_diff_srv_rate` |
| 7 | `dst_host_same_srv_rate` |
| 8 | `dst_host_count` |
| 9 | `dst_host_same_src_port_rate` |
| 10 | `logged_in` |

This provides an interpretable view of which network-traffic characteristics contribute most strongly to the model's decisions.

SHAP artifacts are stored under:

```text
python_pipeline/models/shap_analysis/
```

---

# 8. Serving — FastAPI

The trained model is exposed through a FastAPI service.

Endpoints:

```text
GET  /health
POST /predict
```

`/predict` accepts a complete 41-feature network traffic record and returns:

- Predicted attack class
- Confidence
- Top three SHAP-driving features

The same preprocessing logic used during training is applied during inference.

This prevents training/serving feature mismatch.

---

# 9. Docker Deployment

The API is containerized using Docker.

Build:

```bash
docker build -t netraapi:latest python_pipeline/
```

Run:

```bash
docker run -p 8000:8000 netraapi:latest
```

Health check:

```text
GET /health
```

The containerized API was tested using 100 sequential prediction requests.

### Docker load-test result

| Metric | Result |
|---|---:|
| Mean | 15.03 ms |
| Median | 13.90 ms |
| P95 | 16.56 ms |
| P99 | 19.79 ms |
| Minimum | 13.46 ms |
| Maximum | 85.95 ms |

The project target was:

```text
< 200 ms
```

The Docker deployment therefore passes the latency target under the tested sequential workload.

---

# 10. Julia Performance Benchmark

The final stage reimplements the tree-based approach in Julia using:

- Julia
- MLJ
- DecisionTree
- BenchmarkTools

The Python preprocessing output is exported to:

```text
julia_benchmark/benchmarks/data/
```

The handoff contains:

```text
X_train_processed.csv
y_train.csv
feature_names.json
handoff_metadata.json
```

The handoff was validated as:

```text
125,973 rows
44 features
5 classes
```

---

## Benchmark

Twenty Random Forest configurations were evaluated using a fixed 80/20 split of `KDDTrain+`.

The configuration with the highest macro-F1 was:

```text
n_trees            = 100
max_depth          = 20
min_samples_leaf   = 1
n_subfeatures      = 22
sampling_fraction  = 0.7
```

Benchmark result:

| Metric | Result |
|---|---:|
| Accuracy | **99.8730%** |
| Macro-F1 | **0.927574** |
| Training time | **15.8049 s** |
| Prediction time | **0.1029 s** |

The final selected model is serialized as:

```text
julia_benchmark/models/netra_random_forest.jls
```

---

## Julia Final Model

The final model was retrained and saved using the benchmark-selected configuration.

Final metrics:

| Metric | Result |
|---|---:|
| Accuracy | **99.8730%** |
| Macro Precision | **0.999278** |
| Macro Recall | **0.889436** |
| Macro-F1 | **0.927574** |
| Weighted-F1 | **0.998699** |

The measured retraining and prediction times for the final model run were:

```text
Training:    18.8876 s
Prediction:   2.4736 s
```

Peak process memory was measured separately using macOS process statistics.

```text
Peak RSS:       ~1.52 GB
Peak footprint: ~1.26 GB
```

Allocated bytes reported by individual benchmark runs are kept separate from peak-memory measurements and are not treated as equivalent metrics.

---

# Project Structure

```text
NETRA/
│
├── data/
│   ├── feature_names.txt
│   └── raw/
│       ├── KDDTrain+.txt
│       └── KDDTest+.txt
│
├── r_analysis/
│   ├── R/
│   └── reports/
│       └── feature_report.html
│
├── python_pipeline/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── load_test.py
│   │
│   ├── config/
│   │   └── feature_config.yaml
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── tune_xgboost.py
│   │   ├── eval_xgboost.py
│   │   ├── final_evaluation.py
│   │   ├── failure_analysis.py
│   │   ├── feature_failure_analysis.py
│   │   ├── feature_ablation.py
│   │   ├── feature_ablation_cv.py
│   │   ├── shap_analysis.py
│   │   ├── prepare_serving_model.py
│   │   └── export_julia_data.py
│   │
│   ├── preprocessing/
│   │   └── preprocessor.py
│   │
│   ├── Dockerfile
│   └── requirements.txt
│
├── julia_benchmark/
│   ├── benchmarks/
│   ├── models/
│   │   └── netra_random_forest.jls
│   ├── reports/
│   │   ├── benchmark_report.md
│   │   ├── benchmark_results.csv
│   │   ├── final_model_metrics.csv
│   │   └── final_model_confusion_matrix.csv
│   └── src/
│       ├── benchmark.jl
│       ├── data.jl
│       ├── model.jl
│       └── train.jl
│
└── README.md
```

---

# Reproducibility

## Python environment

Python dependencies are specified in:

```text
python_pipeline/requirements.txt
```

Install them with:

```bash
pip install -r python_pipeline/requirements.txt
```

Run the relevant pipeline scripts from the repository root.

---

## Julia environment

The Julia project is isolated in:

```text
julia_benchmark/
```

Activate it with:

```bash
julia --project=julia_benchmark
```

The environment is locked through:

```text
Project.toml
Manifest.toml
```

---

# Key Design Decisions

### Why R?

R provides a dedicated statistical-analysis stage before model development.

This makes feature selection and transformation decisions explicit rather than embedding them directly inside the model-training code.

### Why Python?

Python provides the main machine-learning ecosystem used by NETRA:

- scikit-learn
- XGBoost
- Optuna
- SHAP
- FastAPI

### Why Julia?

Julia is used as a dedicated performance-benchmarking environment rather than as another production-serving dependency.

This keeps the production path simple while allowing the project to investigate compiled numerical performance separately.

### Why no feature scaling?

The primary models are tree-based models, which do not require conventional feature scaling for split selection.

NETRA therefore uses targeted transformations such as `log1p` rather than applying a generic standardization step.

### Why use KDDTest+ only at the end?

Repeatedly optimizing against the official test set would contaminate the evaluation.

`KDDTest+` is therefore treated as a frozen final evaluation dataset.

---

# Limitations

NETRA has several important limitations.

### Dataset limitations

NSL-KDD is a benchmark dataset rather than a representation of modern production network traffic.

### Distribution shift

The largest observed limitation is the difference between training/CV behavior and official test-set behavior, particularly for R2L and U2R attacks.

### Rare classes

U2R attacks are extremely underrepresented, making reliable generalization difficult.

### Generalization

Near-perfect cross-validation results should not be interpreted as evidence of equivalent real-world intrusion-detection performance.

### Deployment scope

The current API is a research/demo inference service rather than a complete production network-monitoring platform.

---

# Future Work

Potential future improvements include:

- Training with more diverse intrusion datasets
- Investigating domain adaptation and distribution-shift handling
- Improving R2L and U2R recall
- Evaluating additional imbalance strategies
- Testing calibration of prediction confidence
- Expanding API monitoring and observability
- Investigating online / streaming inference
- Evaluating performance on additional intrusion-detection datasets
- Comparing additional model families

---

# Results at a Glance

```text
                    NETRA

Statistical Analysis
        R
        │
        ▼
Feature Engineering
        Python
        │
        ▼
XGBoost + Optuna
        │
        ├── SHAP Explainability
        │
        └── FastAPI + Docker
                  │
                  ▼
             <200 ms target
                  PASS

Official KDDTest+
        │
        ▼
79.4846% accuracy
0.621465 macro-F1
0.973011 binary ROC-AUC

Julia Benchmark
        │
        ▼
99.8730% accuracy
0.927574 macro-F1
15.8049 s training
0.1029 s prediction
```

---

# Conclusion

NETRA is designed as an end-to-end investigation into network intrusion detection rather than simply a classifier benchmark.

The project demonstrates the complete path from:

**statistical analysis → feature engineering → model development → tuning → evaluation → failure analysis → explainability → API deployment → containerization → performance benchmarking.**

The strongest lesson from the project is that extremely high validation performance does not guarantee strong generalization to a shifted test distribution.

NETRA therefore treats its failures as part of the result: the model performs strongly on several major traffic classes, while R2L and U2R expose the limitations of learning from an imbalanced and distribution-shifted benchmark dataset.

---

## Author

**Rits**

Built as a machine-learning and systems engineering project combining statistical analysis, cybersecurity, explainable AI, API deployment, and performance engineering.
