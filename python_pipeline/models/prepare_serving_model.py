from pathlib import Path
import json
import pickle
import sys

import pandas as pd
import yaml
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

SHAP_DIR = (
    PROJECT_ROOT
    / "python_pipeline"
    / "models"
    / "shap_analysis"
)

SERVING_DIR = (
    PROJECT_ROOT
    / "python_pipeline"
    / "models"
    / "serving"
)

SERVING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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

ALL_COLUMNS = FEATURE_NAMES + [
    "label",
    "difficulty",
]


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


CLASS_NAMES = [
    "normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


def load_training_data():
    print(f"Loading {TRAIN_PATH}")

    df = pd.read_csv(
        TRAIN_PATH,
        header=None,
        names=ALL_COLUMNS,
    )

    if df.shape != (125973, 43):
        raise ValueError(
            f"Unexpected training shape: {df.shape}"
        )

    df["attack_class"] = df["label"].map(
        ATTACK_CLASS_MAP
    )

    if df["attack_class"].isna().any():
        unknown = sorted(
            df.loc[
                df["attack_class"].isna(),
                "label",
            ].unique()
        )

        raise ValueError(
            f"Unknown labels: {unknown}"
        )

    return df


def load_best_params():
    with open(
        BEST_PARAMS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        result = json.load(f)

    return result["best_params"]


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
        random_state=42,
        n_jobs=-1,
    )


def main():
    print("Preparing NETRA serving model")

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    df = load_training_data()

    X_raw = df[FEATURE_NAMES]

    y = df["attack_class"].map(
        CLASS_TO_ID
    ).to_numpy()

    print("Fitting NETRApreprocessor")

    preprocessor = NETRApreprocessor(
        CONFIG_PATH
    )

    X = preprocessor.fit_transform(
        X_raw
    )

    print(
        f"Processed features: {X.shape[1]}"
    )

    params = load_best_params()

    model = create_model(
        params
    )

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    weights *= float(
        params["weight_multiplier"]
    )

    print("Training final serving model")

    model.fit(
        X,
        y,
        sample_weight=weights,
    )

    print("Saving artifacts")

    model_path = (
        SERVING_DIR
        / "model.pkl"
    )

    preprocessor_path = (
        SERVING_DIR
        / "preprocessor.pkl"
    )

    with open(
        model_path,
        "wb",
    ) as f:
        pickle.dump(
            model,
            f,
        )

    with open(
        preprocessor_path,
        "wb",
    ) as f:
        pickle.dump(
            preprocessor,
            f,
        )

    # SHAP uses the processed feature names.
    processed_feature_names_path = (
        SHAP_DIR
        / "processed_feature_names.json"
    )

    if not processed_feature_names_path.exists():
        raise FileNotFoundError(
            "SHAP processed feature names not found: "
            f"{processed_feature_names_path}"
        )

    with open(
        processed_feature_names_path,
        "r",
        encoding="utf-8",
    ) as f:
        processed_feature_names = json.load(f)

    metadata = {
        "model_type": "weighted_xgboost",
        "classes": CLASS_NAMES,
        "raw_feature_count": len(
            FEATURE_NAMES
        ),
        "processed_feature_count": len(
            processed_feature_names
        ),
        "raw_features": FEATURE_NAMES,
        "processed_features":
            processed_feature_names,
        "random_state": 42,
        "weight_multiplier":
            params["weight_multiplier"],
        "model_parameters": {
            key: value
            for key, value in params.items()
            if key != "weight_multiplier"
        },
    }

    metadata_path = (
        SERVING_DIR
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("Serving artifacts created:")
    print(f"  {model_path}")
    print(f"  {preprocessor_path}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise