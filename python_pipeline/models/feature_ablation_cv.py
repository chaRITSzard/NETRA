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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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
    / "feature_ablation_cv"
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


# Attack mapping
ATTACK_CLASS_MAP = {
    "normal": "normal",

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

    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

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


# Candidate configurations
CANDIDATES = {
    "current_policy": [],

    "restore_dst_host_srv_count": [
        "dst_host_srv_count",
    ],
}

# Tuned weighted XGBoost configuration
MODEL_PARAMS = {
    "n_estimators": 215,
    "max_depth": 10,
    "learning_rate": 0.07625952012379998,
    "subsample": 0.9631804864850304,
    "colsample_bytree": 0.9649627446231271,
}

WEIGHT_MULTIPLIER = 1.5868169724223125

RANDOM_STATE = 42

N_SPLITS = 5


# Dataset
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

# Preprocessor configuration
def build_config(base_config, restored_features):
    config = yaml.safe_load(
        yaml.safe_dump(base_config)
    )

    config["features"]["drop"] = [
        feature
        for feature in config["features"]["drop"]
        if feature not in restored_features
    ]

    return config


def create_temp_config(base_config, restored_features, name):
    config = build_config(
        base_config,
        restored_features,
    )

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

# CV experiment
def run_candidate(
    candidate_name,
    restored_features,
    df,
    y,
    base_config,
    cv,
):
    print("\n" + "=" * 70)
    print(f"CANDIDATE: {candidate_name}")
    print("=" * 70)

    if restored_features:
        print(
            "Restored:",
            ", ".join(restored_features),
        )
    else:
        print("Restored: none")

    config_path = create_temp_config(
        base_config,
        restored_features,
        candidate_name,
    )

    X_raw = df[FEATURE_NAMES]

    fold_rows = []

    # Accumulate out-of-fold predictions.
    oof_predictions = np.full(
        len(df),
        -1,
        dtype=int,
    )

    total_start = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X_raw, y),
        start=1,
    ):
        print(f"\nFold {fold}/{N_SPLITS}")

        X_train_raw = X_raw.iloc[train_idx]
        X_valid_raw = X_raw.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # --------------------------------------------------------
        # Fresh preprocessor for every fold.
        # --------------------------------------------------------

        preprocessor = NETRApreprocessor(
            config_path
        )

        X_train = preprocessor.fit_transform(
            X_train_raw
        )

        X_valid = preprocessor.transform(
            X_valid_raw
        )

        print(
            f"Processed features: {X_train.shape[1]}"
        )

        # --------------------------------------------------------
        # Weighted training
        # --------------------------------------------------------

        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        weights *= WEIGHT_MULTIPLIER

        model = create_model()

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
        )

        training_time = (
            time.perf_counter() - start
        )

        predictions = model.predict(
            X_valid
        ).astype(int)

        oof_predictions[valid_idx] = predictions

        accuracy = accuracy_score(
            y_valid,
            predictions,
        )

        macro_f1 = f1_score(
            y_valid,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_valid,
            predictions,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            y_valid,
            predictions,
            labels=list(range(5)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        row = {
            "candidate": candidate_name,
            "fold": fold,
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
        }

        fold_rows.append(row)

        print(
            f"Accuracy={accuracy:.6f} | "
            f"Macro-F1={macro_f1:.6f} | "
            f"R2L F1={report['R2L']['f1-score']:.6f} | "
            f"U2R F1={report['U2R']['f1-score']:.6f}"
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    fold_df = pd.DataFrame(fold_rows)

    # ------------------------------------------------------------
    # Aggregate fold statistics
    # ------------------------------------------------------------

    aggregate = {
        "candidate": candidate_name,
        "processed_features":
            int(fold_df["processed_features"].iloc[0]),

        "accuracy_mean":
            fold_df["accuracy"].mean(),
        "accuracy_std":
            fold_df["accuracy"].std(ddof=1),

        "macro_f1_mean":
            fold_df["macro_f1"].mean(),
        "macro_f1_std":
            fold_df["macro_f1"].std(ddof=1),

        "weighted_f1_mean":
            fold_df["weighted_f1"].mean(),
        "weighted_f1_std":
            fold_df["weighted_f1"].std(ddof=1),

        "R2L_precision_mean":
            fold_df["R2L_precision"].mean(),
        "R2L_recall_mean":
            fold_df["R2L_recall"].mean(),
        "R2L_f1_mean":
            fold_df["R2L_f1"].mean(),

        "U2R_precision_mean":
            fold_df["U2R_precision"].mean(),
        "U2R_recall_mean":
            fold_df["U2R_recall"].mean(),
        "U2R_f1_mean":
            fold_df["U2R_f1"].mean(),

        "total_runtime_seconds":
            total_time,
    }

    # ------------------------------------------------------------
    # OOF metrics
    # ------------------------------------------------------------

    if np.any(oof_predictions < 0):
        raise RuntimeError(
            "OOF predictions incomplete."
        )

    oof_accuracy = accuracy_score(
        y,
        oof_predictions,
    )

    oof_macro_f1 = f1_score(
        y,
        oof_predictions,
        average="macro",
        zero_division=0,
    )

    oof_weighted_f1 = f1_score(
        y,
        oof_predictions,
        average="weighted",
        zero_division=0,
    )

    oof_report = classification_report(
        y,
        oof_predictions,
        labels=list(range(5)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    aggregate.update(
        {
            "oof_accuracy": oof_accuracy,
            "oof_macro_f1": oof_macro_f1,
            "oof_weighted_f1": oof_weighted_f1,

            "oof_R2L_precision":
                oof_report["R2L"]["precision"],
            "oof_R2L_recall":
                oof_report["R2L"]["recall"],
            "oof_R2L_f1":
                oof_report["R2L"]["f1-score"],

            "oof_U2R_precision":
                oof_report["U2R"]["precision"],
            "oof_U2R_recall":
                oof_report["U2R"]["recall"],
            "oof_U2R_f1":
                oof_report["U2R"]["f1-score"],
        }
    )

    return fold_df, aggregate

# Main
def main():
    print("=" * 70)
    print("NETRA — Step 1b: Feature Ablation 5-Fold CV")
    print("=" * 70)

    print(
        "\nKDDTest+ is NOT loaded."
    )

    # ------------------------------------------------------------
    # Load configuration
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

    y = df["attack_class"].map(
        CLASS_TO_ID
    ).to_numpy()

    print(
        f"\nRows: {len(df)}"
    )

    # ------------------------------------------------------------
    # Fixed 5-fold stratified CV
    # ------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_fold_results = []
    all_aggregate_results = []

    # ------------------------------------------------------------
    # Run candidates
    # ------------------------------------------------------------

    for candidate_name, restored_features in CANDIDATES.items():

        fold_df, aggregate = run_candidate(
            candidate_name= candidate_name,
            restored_features=restored_features,
            df=df,
            y=y,
            base_config=base_config,
            cv=cv,
        )

        all_fold_results.append(fold_df)
        all_aggregate_results.append(aggregate)

    # ------------------------------------------------------------
    # Combine results
    # ------------------------------------------------------------

    folds_df = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    aggregate_df = pd.DataFrame(
        all_aggregate_results
    )

    # ------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------

    folds_path = (
        OUTPUT_DIR
        / "ablation_cv_folds.csv"
    )

    aggregate_path = (
        OUTPUT_DIR
        / "ablation_cv_summary.csv"
    )

    folds_df.to_csv(
        folds_path,
        index=False,
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Remove temporary configs
    # ------------------------------------------------------------

    for path in OUTPUT_DIR.glob("_temp_*.yaml"):
        path.unlink()

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("5-FOLD CV SUMMARY")
    print("=" * 70)

    summary_columns = [
        "candidate",
        "processed_features",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "oof_accuracy",
        "oof_macro_f1",
        "oof_R2L_recall",
        "oof_R2L_f1",
        "oof_U2R_recall",
        "oof_U2R_f1",
    ]

    print(
        aggregate_df[summary_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ------------------------------------------------------------
    # Determine development-side candidate
    # ------------------------------------------------------------

    ranking = aggregate_df.sort_values(
        by=[
            "oof_macro_f1",
            "oof_R2L_f1",
            "oof_U2R_f1",
        ],
        ascending=False,
    )

    ranking_path = (
        OUTPUT_DIR
        / "ablation_cv_ranking.csv"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    winner = ranking.iloc[0]

    print("\n" + "=" * 70)
    print("DEVELOPMENT-SIDE RESULT")
    print("=" * 70)

    print(
        f"Top candidate by OOF Macro-F1: "
        f"{winner['candidate']}"
    )

    print(
        f"OOF accuracy: "
        f"{winner['oof_accuracy']:.6f}"
    )

    print(
        f"OOF Macro-F1: "
        f"{winner['oof_macro_f1']:.6f}"
    )

    print(
        f"OOF R2L recall: "
        f"{winner['oof_R2L_recall']:.6f}"
    )

    print(
        f"OOF U2R recall: "
        f"{winner['oof_U2R_recall']:.6f}"
    )

    print("\nSaved:")
    print(f"  {folds_path}")
    print(f"  {aggregate_path}")
    print(f"  {ranking_path}")

    print("\n" + "=" * 70)
    print("IMPORTANT")
    print("=" * 70)
    print(
        "This is a training-side validation experiment."
    )
    print(
        "KDDTest+ remains frozen and must not be used "
        "to select between these candidates."
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