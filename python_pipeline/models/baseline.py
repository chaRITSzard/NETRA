from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from python_pipeline.preprocessing.preprocessor import NETRApreprocessor
#PATHS
ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "KDDTrain+.txt"
CONFIG_PATH = ROOT / "python_pipeline" / "config" / "feature_config.yaml"

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

RAW_COLUMNS = FEATURE_NAMES + ["label", "difficulty"]

#5-Class Mapping
ATTACK_CLASS_MAP = {
#normal
    "normal" : "normal",
#Denial of Service(DoS)
    "back" : "DoS",
    "land" : "DoS",
    "neptune" : "DoS",
    "pod" : "DoS",
    "smurf" : "DoS",
    "teardrop" : "DoS",
    "mailbomb" : "DoS",
    "processtable" : "DoS",
    "udpstorm" : "DoS",
    "apache2" : "DoS",
    "worm" : "DoS",
#Probe Attack
    "satan" : "Probe",
    "ipsweep" : "Probe",
    "nmap" : "Probe",
    "portsweep" : "Probe",
    "mscan" : "Probe",
    "saint" : "Probe",
#R2L
    "guess_passwd" : "R2L",
    "ftp_write" : "R2L",
    "imap" : "R2L",
    "phf" : "R2L",
    "multihop" : "R2L",
    "warezmaster" : "R2L",
    "warezclient" : "R2L",
    "spy" : "R2L",
    "named" : "R2L",
    "sendmail" : "R2L",
    "snmpgetattack" : "R2L",
    "snmpguess" : "R2L",
    "xlock" : "R2L",
    "xsnoop" : "R2L",
    "httptunnel" : "R2L",
    "sqlattack" : "R2L",
    "ps" : "R2L",
#U2R
    "buffer_overflow" : "U2R",
    "loadmodule" : "U2R",
    "rootkit" : "U2R",
    "perl" : "U2R",
    "xterm" : "U2R"
}

CLASS_NAMES = ["normal", "DoS", "Probe", "R2L", "U2R"]

CLASS_TO_ID = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

#DATA LOADING
def load_training_data():
    print("Loading KDDTrain+...")

    df = pd.read_csv(
        TRAIN_PATH,
        header=None,
        names=RAW_COLUMNS
    )

    print(f"No. of Rows:   {len(df)}")
    print(f"No. of Columns:   {len(df.columns)}")

    if len(df) != 125973:
        raise ValueError(
            f"Unexpected KDDTrain+ row count: {len(df)}"
        )

    if len(df.columns) != 43:
        raise ValueError(
            f"Unexpected KDDTrain+ column count: {len(df.columns)}"
        )

    df["attack_class"] = df["label"].map(ATTACK_CLASS_MAP)

    unknown_labels = sorted(
        df.loc[df["attack_class"].isna(), "label"].unique()
    )

    if unknown_labels:
        raise ValueError(
            f"Unmapped attack labels found: {unknown_labels}"
    )

    X = df[FEATURE_NAMES].copy()
    y = df["attack_class"].copy()

    print("\n5-Class Distribution:")
    print(y.value_counts().reindex(CLASS_NAMES))

    return X, y

#MODEL FACTORY
def create_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

def create_XGBoost():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

def create_weighted_XGBoost():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

#CROSS-VALIDATION

def run_model_cv(model_name, model_factory, X, y):
    print("\n")
    print("="*70)
    print(f"{model_name} - 5-FOLD STRATIFIED CV")
    print("="*70)

    y_encoded = y.map(CLASS_TO_ID).to_numpy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    all_true = []
    all_pred = []

    total_start = time.perf_counter()

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y_encoded),
        start=1,
    ):
        print(f"\n--- Fold {fold}/5 ---")

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()

        y_train = y_encoded[train_idx]
        y_val = y_encoded[val_idx]


        preprocessor = NETRApreprocessor(CONFIG_PATH)

        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_val_preprocessed = preprocessor.transform(X_val)

        print(
            f"Processed Shapes: "
            f"{X_train_preprocessed.shape} train/"
            f"{X_val_preprocessed.shape} validation"
        )

        model = model_factory()

        start = time.perf_counter()

        if model_name == "XGBoost (Weighted)":
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

            model.fit(
                X_train_preprocessed,
                y_train,
                sample_weight=sample_weights,
            )
        else:
            model.fit(
                X_train_preprocessed,
                y_train,
            )

        training_time = time.perf_counter() - start

        predictions = model.predict(X_val_preprocessed)

        accuracy = accuracy_score(
            y_val,
            predictions
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_val,
            predictions,
            average="weighted",
            zero_division=0,
        )

        print(f"Accuracy:   {accuracy:.6f}")
        print(f"Macro-F1:   {macro_f1:.6f}")
        print(f"Weighted-F1:   {weighted_f1:.6f}")
        print(f"Training:   {training_time:.3f}s")

        fold_results.append(
            {
                "fold":fold,
                "accuracy":accuracy,
                "macro_f1":macro_f1,
                "weighted_f1":weighted_f1,
                "training_time_sec":training_time, 
            }
        )

        all_true.extend(y_val)
        all_pred.extend(predictions)

    total_time = time.perf_counter() - total_start
    results_df = pd.DataFrame(fold_results)

    print("\n")
    print("-"*70)
    print(f"{model_name} - CV SUMMARY")
    print("-"*70)

    for metric in [
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]:
        mean = results_df[metric].mean()
        std = results_df[metric].std()

        print(
            f"{metric:15s}: "
            f"{mean:.6f} ± {std:.6f}"
        )

    print(
        f"Total CV time: {total_time:.3f}"
    )

    #AGGREGATED CONFUSION MATRIX
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            cm,
            index = [f"Actual {c}" for c in CLASS_NAMES],
            columns = [f"Pred {c}" for c in CLASS_NAMES]
        )
    )

    #PER-CLASS METRICS
    print("\nPer-Class Metrics:")

    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0
    )
    print(report)

    return {
        "model": model_name,
        "fold_results": results_df,
        "confusion_matrix": cm,
        "classification_report": report,
    }

#main
def main():
    X, y = load_training_data()

    rf_results = run_model_cv(
        "Random Forest",
        create_random_forest,
        X,
        y,
    )

    xgb_results = run_model_cv(
        "XGBoost",
        create_XGBoost,
        X,
        y,
    )
    weighted_xgb_results = run_model_cv(
        "XGBoost (Weighted)",
        create_weighted_XGBoost,
        X,
        y,
    )

    print("\n")
    print("=" * 70)
    print("NETRA BASELINE COMPLETE")
    print("=" * 70)

    print("\nModels:")
    print("  ✓ Random Forest")
    print("  ✓ XGBoost")

    print("\nKDDTest+ was NOT used.")
    print("These are training-set cross-validation results only.")


if __name__ == "__main__":
    main()