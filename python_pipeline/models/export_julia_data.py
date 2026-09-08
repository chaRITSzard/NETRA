from pathlib import Path
import json

import pandas as pd

from python_pipeline.preprocessing.preprocessor import NETRApreprocessor


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/raw/KDDTrain+.txt"
CONFIG_PATH = ROOT / "python_pipeline/config/feature_config.yaml"
OUTPUT_DIR = ROOT / "julia_benchmark/benchmarks/data"

FEATURE_NAMES_PATH = ROOT / "data/feature_names.txt"

ATTACK_MAP = {
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


def load_train():
    feature_names = FEATURE_NAMES_PATH.read_text().splitlines()

    columns = feature_names + ["label", "difficulty"]

    return pd.read_csv(
        TRAIN_PATH,
        names=columns,
        header=None,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train()

    df["attack_class"] = df["label"].map(ATTACK_MAP)

    if df["attack_class"].isna().any():
        unknown = sorted(df.loc[df["attack_class"].isna(), "label"].unique())
        raise ValueError(f"Unknown attack labels: {unknown}")

    raw_features = df[FEATURE_NAMES_PATH.read_text().splitlines()]

    preprocessor = NETRApreprocessor(CONFIG_PATH)
    X = preprocessor.fit_transform(raw_features)

    y = df["attack_class"]

    expected_classes = ["normal", "DoS", "Probe", "R2L", "U2R"]

    if X.shape != (125973, 44):
        raise ValueError(f"Unexpected processed shape: {X.shape}")

    if y.isna().any():
        raise ValueError("Missing attack classes after mapping")

    if sorted(y.unique()) != sorted(expected_classes):
        raise ValueError(f"Unexpected classes: {sorted(y.unique())}")

    X.to_csv(
        OUTPUT_DIR / "X_train_processed.csv",
        index=False,
    )

    y.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False,
    )

    (OUTPUT_DIR / "feature_names.json").write_text(
        json.dumps(X.columns.tolist(), indent=2)
    )

    metadata = {
        "dataset": "NSL-KDD",
        "source": "KDDTrain+",
        "rows": int(X.shape[0]),
        "processed_features": int(X.shape[1]),
        "classes": expected_classes,
        "preprocessing_config": str(
            CONFIG_PATH.relative_to(ROOT)
        ),
        "feature_order_file": "feature_names.json",
        "test_set_used": False,
    }

    (OUTPUT_DIR / "handoff_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("Julia handoff created successfully")
    print(f"Rows: {X.shape[0]}")
    print(f"Processed features: {X.shape[1]}")
    print(f"Classes: {expected_classes}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()