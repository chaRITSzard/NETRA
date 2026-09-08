# NETRA Julia Benchmark Report

## 1. Overview

This report documents Phase 5 of NETRA (Network Threat Recognition & Analysis).

The Julia layer provides a Julia-native tree-ensemble implementation for
performance benchmarking. It is a benchmarking/comparison layer and does not
replace the Python XGBoost model used by the NETRA serving API.

KDDTest+ was not used during Julia model selection or benchmarking.

## 2. Environment

| Component | Version |
|---|---:|
| Julia | 1.12.7 |
| MLJ | 0.23.3 |
| DecisionTree | 0.12.4 |
| BenchmarkTools | 1.8.0 |

## 3. Data and preprocessing

The Julia benchmark consumes the processed representation exported by the
Python pipeline.

Source dataset:

- NSL-KDD KDDTrain+
- 125,973 training rows
- 44 processed features
- 5 attack classes

Classes:

- normal
- DoS
- Probe
- R2L
- U2R

The processed representation preserves the frozen NETRA Python preprocessing
policy, including:

- one-hot encoding for protocol_type and flag
- frequency encoding for service
- log1p transformation for duration, src_bytes, and dst_bytes
- frozen feature-drop policy
- 44 processed features

No KDDTest+ data was exported to the Julia benchmark.

## 4. Julia model

The Julia implementation uses DecisionTree.jl through the MLJ interface.

Model:

`RandomForestClassifier`

Fixed parameters:

| Parameter | Value |
|---|---:|
| sampling_fraction | 0.7 |
| rng | 42 |
| min_samples_split | default |
| min_purity_increase | default |
| feature_importance | default |

The benchmark varied:

- n_trees
- max_depth
- min_samples_leaf
- n_subfeatures

## 5. Benchmark methodology

A fixed 80/20 validation split was created from KDDTrain+ using random seed 42.

Twenty configurations were evaluated.

For each configuration, the benchmark recorded:

- training wall-clock time
- prediction wall-clock time
- allocated bytes during training
- allocated bytes during prediction
- validation accuracy
- validation macro-F1

The benchmark used one fixed validation split rather than repeated
cross-validation.

The allocated-byte measurements are not interpreted as peak memory.

Peak resident memory was measured separately for the final model process using
the macOS `/usr/bin/time -l` measurement.

## 6. Benchmark winner

The configuration selected by validation macro-F1 was:

| Parameter | Value |
|---|---:|
| n_trees | 100 |
| max_depth | 20 |
| min_samples_leaf | 1 |
| n_subfeatures | 22 |
| sampling_fraction | 0.7 |
| rng | 42 |

Results:

| Metric | Result |
|---|---:|
| Accuracy | 99.8729907% |
| Macro-F1 | 0.9275737 |
| Training time | 15.8049 s |
| Prediction time | 0.1029 s |

This configuration achieved the highest macro-F1 among the 20 tested
configurations.

## 7. Benchmark observations

Increasing the number of trees did not improve macro-F1 after the selected
100-tree configuration on this validation split.

For example:

| Configuration | Accuracy | Macro-F1 | Training |
|---|---:|---:|---:|
| 100 trees, depth 20, 22 features | 99.8730% | 0.927574 | 15.8049 s |
| 200 trees, depth 20, 22 features | 99.8770% | 0.917376 | 31.1015 s |
| 400 trees, depth 20, 22 features | 99.8770% | 0.917376 | 62.9307 s |
| 600 trees, depth 20, 22 features | 99.8770% | 0.917376 | 95.6542 s |

The results indicate that additional trees increased training cost without
improving macro-F1 on the fixed validation split.

Feature subsampling also affected the quality/performance trade-off. The
selected configuration used 22 of the 44 processed features per split.

## 8. Final Julia model

The final Julia model was retrained using the selected configuration.

Validation results:

| Metric | Result |
|---|---:|
| Accuracy | 99.8730% |
| Macro precision | 0.999278 |
| Macro recall | 0.889436 |
| Macro-F1 | 0.927574 |
| Weighted-F1 | 0.998699 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 0.998068 | 0.999479 | 0.998773 | 13,439 |
| DoS | 0.999677 | 0.999354 | 0.999515 | 9,284 |
| Probe | 0.998237 | 0.995604 | 0.996919 | 2,275 |
| R2L | 0.989011 | 0.952381 | 0.970350 | 189 |
| U2R | 0.800000 | 0.500000 | 0.615385 | 8 |

The U2R validation support is only eight samples, so its recall estimate is
particularly sensitive to individual observations.

### Confusion matrix

Rows represent actual classes and columns represent predicted classes.

| Actual / Predicted | normal | DoS | Probe | R2L | U2R |
|---|---:|---:|---:|---:|---:|
| normal | 13,434 | 3 | 2 | 0 | 0 |
| DoS | 3 | 9,280 | 1 | 0 | 0 |
| Probe | 9 | 1 | 2,265 | 0 | 0 |
| R2L | 9 | 0 | 0 | 180 | 0 |
| U2R | 4 | 0 | 0 | 0 | 4 |

## 9. Final model execution measurements

The final model training run recorded:

- Training time: 18.8876 s
- Prediction time: 2.4736 s

Peak resident set size for the Julia process:

- 1,517,322,240 bytes
- approximately 1.52 GB
- approximately 1.41 GiB

Peak RSS represents the memory footprint of the Julia process during the
measurement and should not be interpreted as the serialized model size alone.

## 10. Python versus Julia

NETRA's production model remains the tuned weighted XGBoost model implemented
in Python.

The Julia implementation uses RandomForestClassifier from DecisionTree.jl
and therefore is not a direct reimplementation of XGBoost.

The Julia layer is intended to provide a performance-oriented comparison of a
native Julia tree ensemble using the same processed feature representation.

Consequently, differences in accuracy, training time, prediction time, and
memory usage should be interpreted as differences between the respective
model implementations rather than as a language-only benchmark.

## 11. Limitations

1. Julia model selection used a fixed 80/20 validation split from KDDTrain+.
2. The Julia benchmark did not use KDDTest+.
3. The Julia model is Random Forest rather than XGBoost.
4. Peak RSS was measured for the final Julia process rather than isolated
   model memory.
5. The benchmark configurations were deliberately selected rather than being
   an exhaustive Cartesian grid.
6. The benchmark used wall-clock measurements for full model training rather
   than repeated full-training microbenchmarks.
7. U2R validation support was only eight samples.

## 12. Reproducibility

From the repository root:

```bash
julia --project=julia_benchmark julia_benchmark/src/benchmark.jl