from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from python_pipeline.preprocessing.preprocessor import NETRApreprocessor

#PATHS
ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "KDDTrain+.txt"
CONFIG_PATH = ROOT / "python_pipeline" / "config" / "feature_config.yaml"

BEST_PARAMS_PATH = ROOT / "python_pipeline" / "models" / "optuna_results" / "best_params.json"

OUTPUT_DIR = ROOT / "python_pipeline" / "models" / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

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
        names=FEATURE_NAMES + ["label", "difficulty"]
    )

    df["attack_class"] = df["label"].map(ATTACK_CLASS_MAP)
    if df["attack_class"].isna().any():
        unknown = sorted(
            df.loc[df["attack_class"].isna(), "label"].astype(str).unique()
        )
        raise ValueError(
            f"Unknown attack labels encountered: {unknown}"
        )
    return df

#SAMPLE WEIGHTING
def make_sample_weights(y, weight_multiplier):
    """
    Reproduce the weighting scheme used by the Optuna objective.

    Normal / DoS / Probe:
        weight = 1

    R2L / U2R:
        weight = 1 + multiplier * (balanced_weight - 1)
    """
    balanced_weight = compute_sample_weight(
        class_weight="balanced",
        y=y
    )
    sample_weights = np.ones_like(
        balanced_weight,
        dtype=float
    )

    minority_mask = y >= CLASS_TO_ID["R2L"]

    sample_weights[minority_mask] = 1.0 + weight_multiplier*(balanced_weight[minority_mask] - 1.0)

    return sample_weights

#MODEL
def build_model(params):
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

#CROSS VALIDATION EVALUATION
def eval_cv(df, params):

    X = df[FEATURE_NAMES].copy()
    y = df["attack_class"].map(CLASS_TO_ID).astype(int)
    y = y.to_numpy()

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_true = []
    all_pred = []
    fold_results = []

    start_time = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        print()
        print("=" * 70)
        print(f"FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # IMPORTANT:
        # Fit preprocessing only on the training portion.
        preprocessor = NETRApreprocessor(CONFIG_PATH)

        X_train_processed = preprocessor.fit_transform(
            X_train
        )

        X_valid_processed = preprocessor.transform(
            X_valid
        )
        # Reproduce Optuna's weighting exactly.
        sample_weights = make_sample_weights(
            y_train,
            float(params["weight_multiplier"]),
        )

        model = build_model(params)

        model.fit(
            X_train_processed,
            y_train,
            sample_weight=sample_weights,
        )

        predictions = model.predict(
            X_valid_processed
        ).astype(int)

        fold_accuracy = accuracy_score(
            y_valid,
            predictions,
        )

        fold_macro_f1 = f1_score(
            y_valid,
            predictions,
            average="macro",
            zero_division=0,
        )

        fold_weighted_f1 = f1_score(
            y_valid,
            predictions,
            average="weighted",
            zero_division=0,
        )

        fold_results.append(
            {
                "fold": fold,
                "accuracy": fold_accuracy,
                "macro_f1": fold_macro_f1,
                "weighted_f1": fold_weighted_f1,
            }
        )

        print(f"Accuracy:    {fold_accuracy:.6f}")
        print(f"Macro-F1:    {fold_macro_f1:.6f}")
        print(f"Weighted-F1: {fold_weighted_f1:.6f}")

        all_true.extend(y_valid.tolist())
        all_pred.extend(predictions.tolist())

    elapsed = time.perf_counter() - start_time

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    return (
        all_true,
        all_pred,
        fold_results,
        elapsed,
    )

#REPORTING
def save_results(
    y_true,
    y_pred,
    fold_results,
    elapsed,
    params,
):
    """Save complete CV evaluation results."""

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    # Fold CSV
    fold_df = pd.DataFrame(fold_results)

    fold_path = OUTPUT_DIR / "cv_fold_results.csv"

    fold_df.to_csv(
        fold_path,
        index=False,
    )

    # Confusion matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_path = OUTPUT_DIR / "confusion_matrix.csv"

    cm_df.to_csv(cm_path)


    # Metrics JSON
    metrics = {
        "evaluation": "5-fold stratified cross-validation",
        "dataset": "KDDTrain+",
        "test_set_used": False,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "runtime_seconds": elapsed,
        "best_parameters": params,
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # Human-readable report
    report_path = OUTPUT_DIR / "evaluation_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NETRA — Tuned XGBoost Evaluation\n")
        f.write("=" * 70 + "\n\n")

        f.write("Evaluation protocol\n")
        f.write("-" * 70 + "\n")
        f.write("Dataset: KDDTrain+\n")
        f.write("Method: Stratified 5-fold cross-validation\n")
        f.write("KDDTest+: NOT USED\n")
        f.write(f"Random state: {RANDOM_STATE}\n\n")

        f.write("Overall metrics\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:        {accuracy:.6f}\n")
        f.write(f"Macro Precision: {macro_precision:.6f}\n")
        f.write(f"Macro Recall:    {macro_recall:.6f}\n")
        f.write(f"Macro-F1:        {macro_f1:.6f}\n")
        f.write(f"Weighted-F1:     {weighted_f1:.6f}\n")
        f.write(f"Runtime:         {elapsed:.3f} seconds\n\n")

        f.write("Per-class metrics\n")
        f.write("-" * 70 + "\n")
        f.write(report)
        f.write("\n")

        f.write("Confusion matrix\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Rows = actual classes; columns = predicted classes\n\n"
        )

        f.write(
            pd.DataFrame(
                cm,
                index=CLASS_NAMES,
                columns=CLASS_NAMES,
            ).to_string()
        )

        f.write("\n\n")
        f.write("Best Optuna parameters\n")
        f.write("-" * 70 + "\n")

        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    print()
    print("=" * 70)
    print("TUNED XGBOOST EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Overall metrics:")
    print(f"Accuracy:        {accuracy:.6f}")
    print(f"Macro Precision: {macro_precision:.6f}")
    print(f"Macro Recall:    {macro_recall:.6f}")
    print(f"Macro-F1:        {macro_f1:.6f}")
    print(f"Weighted-F1:     {weighted_f1:.6f}")
    print(f"Runtime:         {elapsed:.3f}s")

    print()
    print("Per-class metrics:")
    print(report)

    print("Confusion matrix:")
    print(
        pd.DataFrame(
            cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )
    )

    print()
    print("Artifacts:")
    print(f"  Fold results:    {fold_path}")
    print(f"  Confusion:       {cm_path}")
    print(f"  Metrics:         {metrics_path}")
    print(f"  Report:          {report_path}")



#MAIN
def main():

    print("=" * 70)
    print("NETRA — TUNED XGBOOST EVALUATION")
    print("=" * 70)

    # Load best Optuna parameters
    with open(BEST_PARAMS_PATH, "r") as f:
        results = json.load(f)

    params = results["best_params"]

    print()
    print("Best Optuna parameters:")

    for key, value in params.items():
        print(f"  {key}: {value}")

    # Load KDDTrain+
    print()
    print(f"Loading: {TRAIN_PATH}")

    df = load_dataset(TRAIN_PATH)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Class distribution:")

    print(
        df["attack_class"]
        .value_counts()
        .reindex(CLASS_NAMES)
        .to_string()
    )

    # CV evaluation
    y_true, y_pred, fold_results, elapsed = eval_cv(
        df,
        params,
    )

    # Save results
    save_results(
        y_true,
        y_pred,
        fold_results,
        elapsed,
        params,
    )


if __name__ == "__main__":
    main()