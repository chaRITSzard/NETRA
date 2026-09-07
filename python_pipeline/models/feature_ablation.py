from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from python_pipeline.preprocessing.preprocessor import NETRApreprocessor

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt"

CONFIG_PATH = (
    PROJECT_ROOT
    / "python_pipeline"
    / "config"
    / "feature_config.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "python_pipeline"
    / "models"
    / "feature_ablation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NSL-KDD schema
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

ALL_COLUMNS = FEATURE_NAMES + ["label", "difficulty"]


# Five-class mapping
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


CLASS_NAMES = ["normal", "DoS", "Probe", "R2L", "U2R"]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

# Ablation configurations
ABLATIONS = {
    "current_policy": [],

    # Feature singled out by the failure analysis:
    # large R2L/U2R train/test shifts.
    "restore_num_compromised": [
        "num_compromised",
    ],

    "restore_dst_host_srv_count": [
        "dst_host_srv_count",
    ],

    # Both individual candidates together.
    "restore_key_dropped": [
        "num_compromised",
        "dst_host_srv_count",
    ],

    # Correlated error-rate group.
    "restore_error_rate_group": [
        "srv_serror_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "srv_rerror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ],

    # All previously dropped features except the constant feature.
    "restore_all_nonconstant_dropped": [
        "num_compromised",
        "srv_serror_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "srv_rerror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "dst_host_srv_count",
    ],
}


# Model parameters

# We use the current tuned candidate so that the ablation isolates
# the effect of the feature policy rather than changing the model.
MODEL_PARAMS = {
    "n_estimators": 215,
    "max_depth": 10,
    "learning_rate": 0.07625952012379998,
    "subsample": 0.9631804864850304,
    "colsample_bytree": 0.9649627446231271,
}


WEIGHT_MULTIPLIER = 1.5868169724223125

RANDOM_STATE = 42

DEV_SIZE = 0.20

# Dataset loading
def load_dataset():
    print(f"Loading: {TRAIN_PATH}")

    df = pd.read_csv(
        TRAIN_PATH,
        header=None,
        names=ALL_COLUMNS,
    )

    if df.shape != (125973, 43):
        raise ValueError(
            f"Unexpected KDDTrain+ shape: {df.shape}"
        )

    df["attack_class"] = df["label"].map(
        ATTACK_CLASS_MAP
    )

    unknown = sorted(
        df.loc[
            df["attack_class"].isna(),
            "label",
        ].unique()
    )

    if unknown:
        raise ValueError(
            f"Unknown attack labels: {unknown}"
        )

    return df

# Build preprocessor with ablation
def create_ablation_preprocessor(
    base_config,
    restored_features,
):
    config = yaml.safe_load(
        yaml.safe_dump(base_config)
    )

    drop_features = config["features"]["drop"]

    config["features"]["drop"] = [
        feature
        for feature in drop_features
        if feature not in restored_features
    ]

    return config


def save_temporary_config(config, name):
    path = OUTPUT_DIR / f"_temp_{name}.yaml"

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    return path

# Model
def create_model():
    return XGBClassifier(
        **MODEL_PARAMS,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

# Single ablation experiment
def run_ablation(
    name,
    restored_features,
    train_df,
    y,
    train_indices,
    dev_indices,
    base_config,
):
    print("\n" + "=" * 70)
    print(f"ABLATION: {name}")
    print("=" * 70)

    if restored_features:
        print(
            "Restored features:",
            ", ".join(restored_features),
        )
    else:
        print("Restored features: none")

    config = create_ablation_preprocessor(
        base_config,
        restored_features,
    )

    temp_config = save_temporary_config(
        config,
        name,
    )

    preprocessor = NETRApreprocessor(
        temp_config
    )

    X_raw = train_df[FEATURE_NAMES]

    X_train_raw = X_raw.iloc[train_indices]
    X_dev_raw = X_raw.iloc[dev_indices]

    y_train = y[train_indices]
    y_dev = y[dev_indices]

    # ------------------------------------------------------------
    # CRITICAL:
    # Fit preprocessing ONLY on the training partition.
    # ------------------------------------------------------------

    start_preprocessing = time.perf_counter()

    X_train = preprocessor.fit_transform(
        X_train_raw
    )

    X_dev = preprocessor.transform(
        X_dev_raw
    )

    preprocessing_time = (
        time.perf_counter()
        - start_preprocessing
    )

    print(
        f"Processed features: {X_train.shape[1]}"
    )

    print(
        f"Preprocessing time: "
        f"{preprocessing_time:.3f}s"
    )

    # ------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    sample_weights *= WEIGHT_MULTIPLIER

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    model = create_model()

    start_training = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    training_time = (
        time.perf_counter()
        - start_training
    )

    # ------------------------------------------------------------
    # Development prediction
    # ------------------------------------------------------------

    start_prediction = time.perf_counter()

    predictions = model.predict(X_dev)

    prediction_time = (
        time.perf_counter()
        - start_prediction
    )

    predictions = predictions.astype(int)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_dev,
        predictions,
    )

    macro_f1 = f1_score(
        y_dev,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_dev,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_dev,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "ablation": name,
        "restored_features": ",".join(
            restored_features
        ),
        "processed_features": X_train.shape[1],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "normal_precision":
            report["normal"]["precision"],
        "normal_recall":
            report["normal"]["recall"],
        "normal_f1":
            report["normal"]["f1-score"],
        "DoS_precision":
            report["DoS"]["precision"],
        "DoS_recall":
            report["DoS"]["recall"],
        "DoS_f1":
            report["DoS"]["f1-score"],
        "Probe_precision":
            report["Probe"]["precision"],
        "Probe_recall":
            report["Probe"]["recall"],
        "Probe_f1":
            report["Probe"]["f1-score"],
        "R2L_precision":
            report["R2L"]["precision"],
        "R2L_recall":
            report["R2L"]["recall"],
        "R2L_f1":
            report["R2L"]["f1-score"],
        "U2R_precision":
            report["U2R"]["precision"],
        "U2R_recall":
            report["U2R"]["recall"],
        "U2R_f1":
            report["U2R"]["f1-score"],
        "training_time_seconds":
            training_time,
        "prediction_time_seconds":
            prediction_time,
    }

    print("\nDevelopment results:")
    print(
        f"Accuracy:   {accuracy:.6f}"
    )
    print(
        f"Macro-F1:   {macro_f1:.6f}"
    )
    print(
        f"Weighted-F1:{weighted_f1:.6f}"
    )

    print("\nPer-class results:")

    for class_name in CLASS_NAMES:
        metrics = report[class_name]

        print(
            f"{class_name:>6} | "
            f"P={metrics['precision']:.6f} "
            f"R={metrics['recall']:.6f} "
            f"F1={metrics['f1-score']:.6f} "
            f"support={int(metrics['support'])}"
        )

    return result

# Main
def main():
    print("=" * 70)
    print("NETRA — Training-Side Feature Ablation")
    print("=" * 70)

    print(
        "\nIMPORTANT: KDDTest+ is NOT loaded or evaluated."
    )

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        base_config = yaml.safe_load(f)

    # ------------------------------------------------------------
    # Load KDDTrain+
    # ------------------------------------------------------------

    df = load_dataset()

    print(
        f"Dataset rows: {len(df)}"
    )

    # ------------------------------------------------------------
    # Convert classes to integer IDs
    # ------------------------------------------------------------

    y = df["attack_class"].map(
        CLASS_TO_ID
    ).to_numpy()

    # ------------------------------------------------------------
    # Single fixed stratified development split
    #
    # The SAME split is used for every ablation so that differences
    # are attributable to the feature configuration rather than
    # different random samples.
    # ------------------------------------------------------------

    indices = np.arange(len(df))

    train_indices, dev_indices = train_test_split(
        indices,
        test_size=DEV_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(
        f"Development split: "
        f"{len(train_indices)} train / "
        f"{len(dev_indices)} development"
    )

    # ------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------

    results = []

    for name, restored_features in ABLATIONS.items():

        result = run_ablation(
            name=name,
            restored_features=restored_features,
            train_df=df,
            y=y,
            train_indices=train_indices,
            dev_indices=dev_indices,
            base_config=base_config,
        )

        results.append(result)

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_path = (
        OUTPUT_DIR
        / "ablation_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Remove temporary configs
    # ------------------------------------------------------------

    for path in OUTPUT_DIR.glob("_temp_*.yaml"):
        path.unlink()

    # ------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------

    ranking = results_df.sort_values(
        by=[
            "macro_f1",
            "R2L_f1",
            "U2R_f1",
        ],
        ascending=False,
    )

    ranking_path = (
        OUTPUT_DIR
        / "ablation_ranking.csv"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    summary_columns = [
        "ablation",
        "processed_features",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "R2L_recall",
        "R2L_f1",
        "U2R_recall",
        "U2R_f1",
    ]

    print(
        ranking[summary_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {ranking_path}")

    print("\n" + "=" * 70)
    print("NEXT DECISION")
    print("=" * 70)
    print(
        "Do NOT change the production feature policy yet."
    )
    print(
        "Inspect ablation_results.csv and compare the "
        "R2L/U2R metrics against the current policy."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise