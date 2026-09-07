from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from python_pipeline.preprocessing.preprocessor import NETRApreprocessor

#PATH
ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "KDDTrain+.txt"
TEST_PATH = ROOT / "data" / "raw" / "KDDTest+.txt"

CONFIG_PATH = ROOT / "python_pipeline" / "config" / "feature_config.yaml"

BEST_PARAMS_PATH = ROOT / "python_pipeline" / "models" / "optuna_results" / "best_params.json"

OUTPUT_DIR = ROOT / "python_pipeline" / "models" / "final_eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

#NSL-KDD SCHEMA
FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CLASS_NAMES = [
    "normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R",
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}

#5-CLASS MAPPING
ATTACK_CLASS_MAP = {
    # Normal
    "normal": "normal",

    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "apache2": "DoS",
    "worm": "DoS",

    # Probe
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    # R2L
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "named": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",
    "sqlattack": "R2L",
    "ps": "R2L",

    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "xterm": "U2R",
}

#DATA LOADING
def load_dataset(path):
    df = pd.read_csv(
        path,
        header=None,
        names=FEATURE_NAMES + ["label", "difficulty"],
    )

    df["attack_class"] = df["label"].map(
        ATTACK_CLASS_MAP
    )

    if df["attack_class"].isna().any():
        unknown = sorted(
            df.loc[
                df["attack_class"].isna(),
                "label",
            ]
            .astype(str)
            .unique()
        )

        raise ValueError(
            f"Unknown attack labels: {unknown}"
        )

    return df

#WEIGHTING
def build_sample_weights(y, weight_multiplier):
    """
    Reproduce the weighting scheme used during Optuna tuning.

    Normal / DoS / Probe:
        weight = 1

    R2L / U2R:
        balanced class weight amplified by weight_multiplier.
    """

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    sample_weights = np.ones_like(
        balanced_weights,
        dtype=float,
    )

    minority_mask = (
        y >= CLASS_TO_ID["R2L"]
    )

    sample_weights[minority_mask] = (
        1.0
        + weight_multiplier
        * (
            balanced_weights[minority_mask]
            - 1.0
        )
    )

    return sample_weights

#MAIN
def main():

    print("=" * 70)
    print("NETRA — FINAL HELD-OUT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Optuna parameters
    # --------------------------------------------------------

    with open(BEST_PARAMS_PATH, "r") as f:
        optuna_results = json.load(f)

    params = optuna_results["best_params"]

    print()
    print("Selected Optuna parameters:")

    for key, value in params.items():
        print(f"  {key}: {value}")

    print()
    print(
        f"Optuna CV Macro-F1: "
        f"{optuna_results['best_macro_f1']:.6f}"
    )

    #LOAD DATASETS
    print()
    print("Loading KDDTrain+...")

    train_df = load_dataset(TRAIN_PATH)

    print(
        f"KDDTrain+ rows: {len(train_df)}"
    )

    print()
    print("Loading KDDTest+...")

    test_df = load_dataset(TEST_PATH)

    print(
        f"KDDTest+ rows: {len(test_df)}"
    )

    #VERIFY TEST SET
    print()
    print("KDDTest+ class distribution:")

    print(
        test_df["attack_class"]
        .value_counts()
        .reindex(CLASS_NAMES)
        .to_string()
    )

    #SEPARATE FEATURES
    X_train = train_df[
        FEATURE_NAMES
    ].copy()

    X_test = test_df[
        FEATURE_NAMES
    ].copy()

    y_train = (
        train_df["attack_class"]
        .map(CLASS_TO_ID)
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test_df["attack_class"]
        .map(CLASS_TO_ID)
        .astype(int)
        .to_numpy()
    )

    #fit() ONLY on KDDTEST+
    print()
    print("Fitting preprocessor on KDDTrain+ only...")

    preprocessor = NETRApreprocessor(
        CONFIG_PATH
    )

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    print(
        f"Processed training features: "
        f"{X_train_processed.shape[1]}"
    )

    print()
    print(
        "Transforming KDDTest+ using the "
        "training-fitted preprocessor..."
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    print(
        f"Processed test features: "
        f"{X_test_processed.shape[1]}"
    )

    #VERIFICATION

    if list(
        X_train_processed.columns
    ) != list(
        X_test_processed.columns
    ):
        raise RuntimeError(
            "Training and test feature columns "
            "do not match."
        )

    if X_test_processed.isna().any().any():
        raise RuntimeError(
            "NaN values found in processed KDDTest+."
        )

    if np.isinf(
        X_test_processed.to_numpy()
    ).any():
        raise RuntimeError(
            "Infinite values found in processed KDDTest+."
        )

    #BUILD MODEL
    model = XGBClassifier(
        n_estimators=int(
            params["n_estimators"]
        ),
        max_depth=int(
            params["max_depth"]
        ),
        learning_rate=float(
            params["learning_rate"]
        ),
        subsample=float(
            params["subsample"]
        ),
        colsample_bytree=float(
            params["colsample_bytree"]
        ),
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print()
    print("=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    print(
        "Training on all KDDTrain+ rows..."
    )

    sample_weights = build_sample_weights(
        y_train,
        float(params["weight_multiplier"]),
    )

    train_start = time.perf_counter()

    model.fit(
        X_train_processed,
        y_train,
        sample_weight=sample_weights,
    )

    train_time = (
        time.perf_counter()
        - train_start
    )

    print(
        f"Training time: {train_time:.3f}s"
    )

    print()
    print("=" * 70)
    print("FINAL KDDTEST+ EVALUATION")
    print("=" * 70)

    print(
        "Evaluating the held-out test set..."
    )

    test_start = time.perf_counter()

    y_pred = model.predict(
        X_test_processed
    ).astype(int)

    y_proba = model.predict_proba(
        X_test_processed
    )

    inference_time = (
        time.perf_counter()
        - test_start
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )


#BINARY ROC-AUC(normal = 0, attack = 1)
    y_test_binary = (
        y_test != CLASS_TO_ID["normal"]
    ).astype(int)

    attack_prob = (
        1.0 - y_proba[
            :, CLASS_TO_ID["normal"]
        ]
    )

    binary_roc_auc = roc_auc_score(
        y_test_binary,
        attack_prob,
    )

    #CLASSIFICATION REPORT
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(
            range(len(CLASS_NAMES))
        ),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(
            range(len(CLASS_NAMES))
        ),
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0,
    )

    #CONFUSION MATRIX
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(
            range(len(CLASS_NAMES))
        ),
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS — KDDTest+")
    print("=" * 70)

    print()
    print(f"Accuracy:        {accuracy:.6f}")
    print(f"Macro Precision: {macro_precision:.6f}")
    print(f"Macro Recall:    {macro_recall:.6f}")
    print(f"Macro-F1:        {macro_f1:.6f}")
    print(f"Weighted-F1:     {weighted_f1:.6f}")
    print(f"Binary ROC-AUC:  {binary_roc_auc:.6f}")

    print()
    print("Per-class metrics:")
    print(report_text)

    print("Confusion matrix:")
    print(cm_df)

    print()
    print(f"Training time:   {train_time:.3f}s")
    print(
        f"Inference time:  "
        f"{inference_time:.3f}s"
    )

    cm_path = OUTPUT_DIR / "confusion_matrix.csv"

    metrics = {
        "evaluation": "final held-out evaluation",
        "train_dataset": "KDDTrain+",
        "test_dataset": "KDDTest+",
        "test_set_usage": "final_evaluation_only",
        "random_state": RANDOM_STATE,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "processed_features": X_train_processed.shape[1],
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "binary_roc_auc": binary_roc_auc,
        "training_time_seconds": train_time,
        "inference_time_seconds": inference_time,
        "best_optuna_macro_f1": optuna_results[
            "best_macro_f1"
        ],
        "best_parameters": params,
        "per_class": report_dict,
    }

    metrics_path = (
        OUTPUT_DIR
        / "final_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    report_path = (
        OUTPUT_DIR
        / "final_evaluation_report.txt"
    )

    with open(report_path, "w") as f:

        f.write(
            "=" * 70 + "\n"
        )
        f.write(
            "NETRA — FINAL KDDTest+ EVALUATION\n"
        )
        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Evaluation protocol\n"
        )
        f.write(
            "-" * 70 + "\n"
        )
        f.write(
            "Training dataset: KDDTrain+\n"
        )
        f.write(
            "Final test dataset: KDDTest+\n"
        )
        f.write(
            "KDDTest+ usage: final evaluation only\n"
        )
        f.write(
            "Preprocessor: fitted on KDDTrain+ only\n"
        )
        f.write(
            "Model: Optuna-tuned weighted XGBoost\n"
        )
        f.write(
            f"Random state: {RANDOM_STATE}\n\n"
        )

        f.write(
            "Overall metrics\n"
        )
        f.write(
            "-" * 70 + "\n"
        )
        f.write(
            f"Accuracy:        {accuracy:.6f}\n"
        )
        f.write(
            f"Macro Precision: {macro_precision:.6f}\n"
        )
        f.write(
            f"Macro Recall:    {macro_recall:.6f}\n"
        )
        f.write(
            f"Macro-F1:        {macro_f1:.6f}\n"
        )
        f.write(
            f"Weighted-F1:     {weighted_f1:.6f}\n"
        )
        f.write(
            f"Binary ROC-AUC:  {binary_roc_auc:.6f}\n"
        )
        f.write(
            f"Training time:   {train_time:.3f}s\n"
        )
        f.write(
            f"Inference time:  {inference_time:.3f}s\n\n"
        )

        f.write(
            "Per-class metrics\n"
        )
        f.write(
            "-" * 70 + "\n"
        )
        f.write(report_text)
        f.write("\n")

        f.write(
            "Confusion matrix\n"
        )
        f.write(
            "-" * 70 + "\n"
        )
        f.write(
            "Rows = actual; columns = predicted\n\n"
        )
        f.write(
            cm_df.to_string()
        )
        f.write("\n\n")

        f.write(
            "Optuna parameters\n"
        )
        f.write(
            "-" * 70 + "\n"
        )

        for key, value in params.items():
            f.write(
                f"{key}: {value}\n"
            )

    print()
    print("=" * 70)
    print("FINAL EVALUATION ARTIFACTS")
    print("=" * 70)

    print(
        f"Confusion matrix: "
        f"{cm_path}"
    )

    print(
        f"Metrics:          "
        f"{metrics_path}"
    )

    print(
        f"Report:           "
        f"{report_path}"
    )

    print()
    print(
        "KDDTest+ has now been used for "
        "the final held-out evaluation."
    )


if __name__ == "__main__":
    main()