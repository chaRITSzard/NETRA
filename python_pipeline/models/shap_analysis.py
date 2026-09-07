from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml
import shap

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from python_pipeline.preprocessing.preprocessor import NETRApreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt"
CONFIG_PATH = (
    PROJECT_ROOT
    / "python_pipeline"
    / "config"
    / "feature_config.yaml"
)

BEST_PARAMS_PATH = (
    PROJECT_ROOT
    / "python_pipeline"
    / "models"
    / "optuna_results"
    / "best_params.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "python_pipeline"
    / "models"
    / "shap_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


RANDOM_STATE = 42

# SHAP can become expensive on the complete dataset.
# We use a fixed, stratified sample for explanation.
SHAP_SAMPLE_SIZE = 5000


def load_dataset():
    print(f"Loading {TRAIN_PATH}")

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


def load_model_params():
    with open(
        BEST_PARAMS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        result = json.load(f)

    params = result["best_params"]

    required = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "weight_multiplier",
    ]

    missing = [
        key
        for key in required
        if key not in params
    ]

    if missing:
        raise ValueError(
            f"Missing model parameters: {missing}"
        )

    return params


def create_model(params):
    return XGBClassifier(
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


def get_shap_matrix(shap_values):
    """
    Normalize SHAP's output across supported SHAP/XGBoost
    multiclass output formats.

    Expected final shape:
        (samples, features, classes)
    """

    if isinstance(shap_values, list):
        # Older SHAP multiclass format:
        # list[class] -> (samples, features)

        return np.stack(
            shap_values,
            axis=-1,
        )

    values = np.asarray(shap_values)

    if values.ndim != 3:
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    # Newer SHAP output can be:
    # (samples, features, classes)
    if values.shape[1] == len(
        SHAP_FEATURE_NAMES
    ):
        return values

    # Or potentially:
    # (samples, classes, features)
    if values.shape[2] == len(
        SHAP_FEATURE_NAMES
    ):
        return np.transpose(
            values,
            (0, 2, 1),
        )

    raise ValueError(
        f"Could not determine SHAP dimensions: "
        f"{values.shape}"
    )


def build_feature_name_list(X):
    return list(X.columns)


def save_global_importance(
    shap_matrix,
    feature_names,
):
    rows = []

    mean_abs_all_classes = np.mean(
        np.abs(shap_matrix),
        axis=2,
    )

    importance = mean_abs_all_classes.mean(
        axis=0
    )

    for feature, value in zip(
        feature_names,
        importance,
    ):
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(value),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    path = (
        OUTPUT_DIR
        / "global_feature_importance.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    return result


def save_class_importance(
    shap_matrix,
    feature_names,
):
    rows = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        values = np.abs(
            shap_matrix[:, :, class_id]
        ).mean(axis=0)

        for feature, value in zip(
            feature_names,
            values,
        ):
            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "mean_abs_shap": float(value),
                }
            )

    result = pd.DataFrame(rows).sort_values(
        [
            "class",
            "mean_abs_shap",
        ],
        ascending=[True, False],
    )

    path = (
        OUTPUT_DIR
        / "class_feature_importance.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    return result


def save_top3_explanations(
    shap_matrix,
    X,
    y,
    predictions,
    feature_names,
):
    rows = []

    for sample_idx in range(
        len(X)
    ):
        predicted_class = int(
            predictions[sample_idx]
        )

        values = shap_matrix[
            sample_idx,
            :,
            predicted_class,
        ]

        top_indices = np.argsort(
            np.abs(values)
        )[::-1][:3]

        for rank, feature_idx in enumerate(
            top_indices,
            start=1,
        ):
            rows.append(
                {
                    "sample_index": sample_idx,
                    "true_class":
                        CLASS_NAMES[
                            int(y[sample_idx])
                        ],
                    "predicted_class":
                        CLASS_NAMES[
                            predicted_class
                        ],
                    "rank": rank,
                    "feature":
                        feature_names[
                            feature_idx
                        ],
                    "feature_value":
                        float(
                            X.iloc[
                                sample_idx,
                                feature_idx
                            ]
                        ),
                    "shap_value":
                        float(
                            values[feature_idx]
                        ),
                    "absolute_shap_value":
                        float(
                            abs(values[feature_idx])
                        ),
                }
            )

    result = pd.DataFrame(rows)

    path = (
        OUTPUT_DIR
        / "top3_prediction_explanations.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    return result


def save_class_top_features(
    class_importance,
):
    rows = []

    for class_name in CLASS_NAMES:
        subset = class_importance[
            class_importance["class"]
            == class_name
        ].head(10)

        for rank, (_, row) in enumerate(
            subset.iterrows(),
            start=1,
        ):
            rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "feature": row["feature"],
                    "mean_abs_shap":
                        row["mean_abs_shap"],
                }
            )

    result = pd.DataFrame(rows)

    path = (
        OUTPUT_DIR
        / "class_top10_features.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    return result


def write_report(
    feature_names,
    global_importance,
    class_importance,
    predictions,
    y,
    params,
    training_time,
    shap_time,
):
    path = (
        OUTPUT_DIR
        / "shap_report.txt"
    )

    lines = []

    lines.append(
        "NETRA — SHAP Explainability Analysis"
    )
    lines.append(
        "SHAP analysis of the frozen weighted XGBoost candidate"
    )
    lines.append("")

    lines.append("DATASET")
    lines.append(
        f"KDDTrain+ rows: 125973"
    )
    lines.append(
        f"SHAP explanation sample: "
        f"{len(y)}"
    )
    lines.append(
        "KDDTest+ used: NO"
    )
    lines.append("")

    lines.append("MODEL")
    lines.append(
        f"n_estimators: "
        f"{int(params['n_estimators'])}"
    )
    lines.append(
        f"max_depth: "
        f"{int(params['max_depth'])}"
    )
    lines.append(
        f"learning_rate: "
        f"{params['learning_rate']}"
    )
    lines.append(
        f"subsample: "
        f"{params['subsample']}"
    )
    lines.append(
        f"colsample_bytree: "
        f"{params['colsample_bytree']}"
    )
    lines.append(
        f"weight_multiplier: "
        f"{params['weight_multiplier']}"
    )
    lines.append("")

    lines.append("MODEL / SHAP TIMING")
    lines.append(
        f"Training time: "
        f"{training_time:.4f}s"
    )
    lines.append(
        f"SHAP calculation time: "
        f"{shap_time:.4f}s"
    )
    lines.append("")

    lines.append("TOP 20 GLOBAL FEATURES")
    lines.append("-" * 40)

    for rank, (_, row) in enumerate(
        global_importance.head(20).iterrows(),
        start=1,
    ):
        lines.append(
            f"{rank:2d}. "
            f"{row['feature']} — "
            f"{row['mean_abs_shap']:.8f}"
        )

    lines.append("")

    lines.append("TOP 10 FEATURES BY CLASS")
    lines.append("-" * 40)

    for class_name in CLASS_NAMES:
        lines.append("")
        lines.append(class_name)

        subset = class_importance[
            class_importance["class"]
            == class_name
        ].head(10)

        for rank, (_, row) in enumerate(
            subset.iterrows(),
            start=1,
        ):
            lines.append(
                f"{rank:2d}. "
                f"{row['feature']} — "
                f"{row['mean_abs_shap']:.8f}"
            )

    lines.append("")

    lines.append("PREDICTION SAMPLE")
    lines.append("-" * 40)

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        count = int(
            (predictions == class_id).sum()
        )

        lines.append(
            f"Predicted {class_name}: "
            f"{count}"
        )

    lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    lines.append(
        "SHAP magnitude represents the contribution "
        "magnitude of a feature toward the model's output."
    )
    lines.append(
        "Global importance is descriptive and does not "
        "establish causal relationships."
    )
    lines.append(
        "Top-3 prediction explanations identify the "
        "three largest absolute SHAP contributions "
        "for the predicted class."
    )
    lines.append(
        "KDDTest+ was not used during this analysis."
    )
    lines.append("")

    lines.append(
        "END OF SHAP ANALYSIS"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
        )

    return path


def main():
    print("NETRA — SHAP Explainability Analysis")
    print()
    print(
        "KDDTest+ will NOT be loaded."
    )

    # Load configuration
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    # Load training data
    df = load_dataset()

    X_raw = df[FEATURE_NAMES]

    y = df["attack_class"].map(
        CLASS_TO_ID
    ).to_numpy()

    # Fit preprocessing on KDDTrain+.
    preprocessor = NETRApreprocessor(
        CONFIG_PATH
    )

    print(
        "Fitting NETRApreprocessor..."
    )

    X = preprocessor.fit_transform(
        X_raw
    )

    feature_names = build_feature_name_list(
        X
    )

    print(
        f"Processed features: "
        f"{X.shape[1]}"
    )

    # Load tuned model parameters.
    params = load_model_params()

    print(
        "Loaded tuned model parameters."
    )

    # Train model on all KDDTrain+.
    weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    weights *= float(
        params["weight_multiplier"]
    )

    model = create_model(params)

    print(
        "Training weighted XGBoost..."
    )

    start = time.perf_counter()

    model.fit(
        X,
        y,
        sample_weight=weights,
    )

    training_time = (
        time.perf_counter()
        - start
    )

    predictions = model.predict(
        X
    ).astype(int)

    # ------------------------------------------------------------
    # Stratified SHAP sample
    # ------------------------------------------------------------

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X),
    )

    sample_indices, _ = train_test_split(
        np.arange(len(X)),
        test_size=(
            len(X) - sample_size
        ),
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_sample = X.iloc[
        sample_indices
    ].reset_index(drop=True)

    y_sample = y[
        sample_indices
    ]

    predictions_sample = predictions[
        sample_indices
    ]

    print(
        f"SHAP sample size: "
        f"{len(X_sample)}"
    )

    # ------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------

    print(
        "Calculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    start = time.perf_counter()

    shap_values = explainer.shap_values(
        X_sample
    )

    shap_time = (
        time.perf_counter()
        - start
    )

    # ------------------------------------------------------------
    # Normalize output
    # ------------------------------------------------------------

    global SHAP_FEATURE_NAMES
    SHAP_FEATURE_NAMES = feature_names

    shap_matrix = get_shap_matrix(
        shap_values
    )

    expected_shape = (
        len(X_sample),
        len(feature_names),
        len(CLASS_NAMES),
    )

    if shap_matrix.shape != expected_shape:
        raise ValueError(
            f"Unexpected normalized SHAP shape: "
            f"{shap_matrix.shape}; "
            f"expected {expected_shape}"
        )

    print(
        f"SHAP shape: "
        f"{shap_matrix.shape}"
    )

    # ------------------------------------------------------------
    # Save explanations
    # ------------------------------------------------------------

    global_importance = (
        save_global_importance(
            shap_matrix,
            feature_names,
        )
    )

    class_importance = (
        save_class_importance(
            shap_matrix,
            feature_names,
        )
    )

    save_class_top_features(
        class_importance
    )

    save_top3_explanations(
        shap_matrix,
        X_sample,
        y_sample,
        predictions_sample,
        feature_names,
    )

    # Save feature names for FastAPI.
    with open(
        OUTPUT_DIR
        / "processed_feature_names.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_names,
            f,
            indent=2,
        )

    # Save model metadata.
    metadata = {
        "model": "weighted_xgboost",
        "random_state": RANDOM_STATE,
        "shap_sample_size":
            len(X_sample),
        "processed_feature_count":
            len(feature_names),
        "weight_multiplier":
            params["weight_multiplier"],
        "kddtest_used": False,
    }

    with open(
        OUTPUT_DIR
        / "shap_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    report_path = write_report(
        feature_names=feature_names,
        global_importance=global_importance,
        class_importance=class_importance,
        predictions=predictions_sample,
        y=y_sample,
        params=params,
        training_time=training_time,
        shap_time=shap_time,
    )

    print()
    print("SHAP analysis complete.")
    print()
    print("Generated:")

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        if path.is_file():
            print(
                f"  - {path.name}"
            )

    print()
    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise