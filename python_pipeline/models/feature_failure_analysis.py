from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTest+.txt"

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
    / "feature_failure_analysis"
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

# Attack-class mapping
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

NUMERIC_FEATURES = [
    "duration",
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

CATEGORICAL_FEATURES = [
    "protocol_type",
    "service",
    "flag",
]

TRANSFORM_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
]

# Utility functions
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path):
    df = pd.read_csv(
        path,
        header=None,
        names=ALL_COLUMNS,
    )

    if df.shape[1] != 43:
        raise ValueError(
            f"{path.name}: expected 43 columns, "
            f"got {df.shape[1]}"
        )

    df["attack_class"] = df["label"].map(ATTACK_CLASS_MAP)

    unknown_labels = sorted(
        df.loc[df["attack_class"].isna(), "label"].unique()
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown attack labels in {path.name}: "
            f"{unknown_labels}"
        )

    return df


def pct(value):
    return f"{value * 100:.4f}%"


def safe_mean(series):
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def safe_median(series):
    if len(series) == 0:
        return np.nan
    return float(series.median())


def safe_std(series):
    if len(series) == 0:
        return np.nan
    return float(series.std())


def safe_min(series):
    if len(series) == 0:
        return np.nan
    return float(series.min())


def safe_max(series):
    if len(series) == 0:
        return np.nan
    return float(series.max())


# Attack distribution
def analyze_attack_distribution(train, test):
    rows = []

    for attack_class in CLASS_NAMES:
        train_count = int(
            (train["attack_class"] == attack_class).sum()
        )

        test_count = int(
            (test["attack_class"] == attack_class).sum()
        )

        train_pct = train_count / len(train)
        test_pct = test_count / len(test)

        rows.append(
            {
                "class": attack_class,
                "train_count": train_count,
                "test_count": test_count,
                "train_pct": train_pct,
                "test_pct": test_pct,
                "percentage_point_shift":
                    (test_pct - train_pct) * 100,
            }
        )

    return pd.DataFrame(rows)

# Numerical feature analysis
def analyze_numeric_features(train, test):
    rows = []

    for attack_class in ["R2L", "U2R"]:
        train_subset = train[
            train["attack_class"] == attack_class
        ]

        test_subset = test[
            test["attack_class"] == attack_class
        ]

        for feature in NUMERIC_FEATURES:
            train_values = pd.to_numeric(
                train_subset[feature],
                errors="coerce",
            )

            test_values = pd.to_numeric(
                test_subset[feature],
                errors="coerce",
            )

            train_mean = safe_mean(train_values)
            test_mean = safe_mean(test_values)

            train_median = safe_median(train_values)
            test_median = safe_median(test_values)

            rows.append(
                {
                    "class": attack_class,
                    "feature": feature,
                    "train_mean": train_mean,
                    "test_mean": test_mean,
                    "mean_difference":
                        test_mean - train_mean,
                    "train_median": train_median,
                    "test_median": test_median,
                    "median_difference":
                        test_median - train_median,
                    "train_std": safe_std(train_values),
                    "test_std": safe_std(test_values),
                    "train_min": safe_min(train_values),
                    "test_min": safe_min(test_values),
                    "train_max": safe_max(train_values),
                    "test_max": safe_max(test_values),
                }
            )

    return pd.DataFrame(rows)

# Dropped feature analysis
def analyze_dropped_features(train, test, drop_features):
    rows = []

    for attack_class in ["R2L", "U2R"]:
        train_subset = train[
            train["attack_class"] == attack_class
        ]

        test_subset = test[
            test["attack_class"] == attack_class
        ]

        for feature in drop_features:
            if feature not in train.columns:
                continue

            train_values = pd.to_numeric(
                train_subset[feature],
                errors="coerce",
            )

            test_values = pd.to_numeric(
                test_subset[feature],
                errors="coerce",
            )

            train_mean = safe_mean(train_values)
            test_mean = safe_mean(test_values)

            train_median = safe_median(train_values)
            test_median = safe_median(test_values)

            rows.append(
                {
                    "class": attack_class,
                    "feature": feature,
                    "train_mean": train_mean,
                    "test_mean": test_mean,
                    "mean_difference":
                        test_mean - train_mean,
                    "train_median": train_median,
                    "test_median": test_median,
                    "median_difference":
                        test_median - train_median,
                    "train_std": safe_std(train_values),
                    "test_std": safe_std(test_values),
                    "train_min": safe_min(train_values),
                    "test_min": safe_min(test_values),
                    "train_max": safe_max(train_values),
                    "test_max": safe_max(test_values),
                    "policy_decision": "DROP",
                }
            )

    return pd.DataFrame(rows)


# Categorical feature analysis
def analyze_categorical_features(train, test):
    rows = []

    for attack_class in ["R2L", "U2R"]:
        train_subset = train[
            train["attack_class"] == attack_class
        ]

        test_subset = test[
            test["attack_class"] == attack_class
        ]

        for feature in CATEGORICAL_FEATURES:
            train_counts = train_subset[feature].value_counts(
                normalize=False
            )

            test_counts = test_subset[feature].value_counts(
                normalize=False
            )

            categories = sorted(
                set(train_counts.index)
                | set(test_counts.index)
            )

            train_total = len(train_subset)
            test_total = len(test_subset)

            for category in categories:
                train_count = int(
                    train_counts.get(category, 0)
                )

                test_count = int(
                    test_counts.get(category, 0)
                )

                train_prop = (
                    train_count / train_total
                    if train_total
                    else 0.0
                )

                test_prop = (
                    test_count / test_total
                    if test_total
                    else 0.0
                )

                if train_count == 0 and test_count > 0:
                    shift_type = "TEST_ONLY"
                elif test_count == 0 and train_count > 0:
                    shift_type = "TRAIN_ONLY"
                elif (
                    abs(test_prop - train_prop) >= 0.10
                ):
                    shift_type = "MAJOR_SHIFT"
                else:
                    shift_type = "STABLE_OR_SMALL_SHIFT"

                rows.append(
                    {
                        "class": attack_class,
                        "feature": feature,
                        "category": category,
                        "train_count": train_count,
                        "test_count": test_count,
                        "train_proportion": train_prop,
                        "test_proportion": test_prop,
                        "percentage_point_shift":
                            (test_prop - train_prop) * 100,
                        "shift_type": shift_type,
                    }
                )

    return pd.DataFrame(rows)

# Transformation analysis
def analyze_transformations(train, test, transform_features):
    rows = []

    for attack_class in ["R2L", "U2R"]:
        train_subset = train[
            train["attack_class"] == attack_class
        ]

        test_subset = test[
            test["attack_class"] == attack_class
        ]

        for feature in transform_features:
            train_raw = pd.to_numeric(
                train_subset[feature],
                errors="coerce",
            ).dropna()

            test_raw = pd.to_numeric(
                test_subset[feature],
                errors="coerce",
            ).dropna()

            train_transformed = np.log1p(train_raw)
            test_transformed = np.log1p(test_raw)

            rows.append(
                {
                    "class": attack_class,
                    "feature": feature,

                    "raw_train_mean":
                        safe_mean(train_raw),
                    "raw_test_mean":
                        safe_mean(test_raw),

                    "raw_train_median":
                        safe_median(train_raw),
                    "raw_test_median":
                        safe_median(test_raw),

                    "raw_train_std":
                        safe_std(train_raw),
                    "raw_test_std":
                        safe_std(test_raw),

                    "log1p_train_mean":
                        safe_mean(train_transformed),
                    "log1p_test_mean":
                        safe_mean(test_transformed),

                    "log1p_train_median":
                        safe_median(train_transformed),
                    "log1p_test_median":
                        safe_median(test_transformed),

                    "log1p_train_std":
                        safe_std(train_transformed),
                    "log1p_test_std":
                        safe_std(test_transformed),

                    "transformation":
                        "log1p",
                }
            )

    return pd.DataFrame(rows)

# Service-specific analysis
def analyze_service_shift(train, test):
    rows = []

    for attack_class in ["R2L", "U2R"]:
        train_subset = train[
            train["attack_class"] == attack_class
        ]

        test_subset = test[
            test["attack_class"] == attack_class
        ]

        train_counts = train_subset["service"].value_counts()
        test_counts = test_subset["service"].value_counts()

        categories = sorted(
            set(train_counts.index)
            | set(test_counts.index)
        )

        for service in categories:
            train_count = int(train_counts.get(service, 0))
            test_count = int(test_counts.get(service, 0))

            train_prop = (
                train_count / len(train_subset)
                if len(train_subset)
                else 0
            )

            test_prop = (
                test_count / len(test_subset)
                if len(test_subset)
                else 0
            )

            rows.append(
                {
                    "class": attack_class,
                    "service": service,
                    "train_count": train_count,
                    "test_count": test_count,
                    "train_proportion": train_prop,
                    "test_proportion": test_prop,
                    "percentage_point_shift":
                        (test_prop - train_prop) * 100,
                    "train_present": train_count > 0,
                    "test_present": test_count > 0,
                }
            )

    return pd.DataFrame(rows)

# Report generation
def write_report(
    train,
    test,
    attack_distribution,
    numeric_analysis,
    dropped_analysis,
    categorical_analysis,
    transformation_analysis,
    service_analysis,
    config,
):
    report_path = OUTPUT_DIR / "feature_failure_report.txt"

    drop_features = config["features"]["drop"]
    keep_features = config["features"]["keep"]
    reencode_features = config["features"]["transform"]

    lines = []

    lines.append("NETRA — Feature-Level Failure Analysis")
    lines.append("=" * 55)
    lines.append("")
    lines.append(
        "Purpose: diagnose whether the Phase 1 feature policy "
        "is associated with the observed R2L/U2R generalization "
        "failure."
    )
    lines.append("")
    lines.append(
        "IMPORTANT: This is diagnostic analysis. "
        "KDDTest+ has already been used for final evaluation, "
        "so these observations must not be used as repeated "
        "test-driven optimization."
    )
    lines.append("")

    lines.append("DATASET")
    lines.append("-" * 55)
    lines.append(f"KDDTrain+ rows: {len(train)}")
    lines.append(f"KDDTest+ rows:  {len(test)}")
    lines.append("Original features: 41")
    lines.append("Processed feature count: 44")
    lines.append("")

    lines.append("CURRENT FEATURE POLICY")
    lines.append("-" * 55)
    lines.append(f"DROP ({len(drop_features)}):")
    lines.extend(f"  - {feature}" for feature in drop_features)
    lines.append("")

    lines.append(f"KEEP ({len(keep_features)}):")
    lines.extend(f"  - {feature}" for feature in keep_features)
    lines.append("")

    lines.append(
        f"RE-ENCODE / TRANSFORM ({len(reencode_features)}):"
    )

    for feature, method in reencode_features.items():
        lines.append(f"  - {feature}: {method}")

    lines.append("")

    lines.append("FIVE-CLASS DISTRIBUTION")
    lines.append("-" * 55)

    for _, row in attack_distribution.iterrows():
        lines.append(
            f"{row['class']}: "
            f"train={row['train_count']} "
            f"({pct(row['train_pct'])}), "
            f"test={row['test_count']} "
            f"({pct(row['test_pct'])}), "
            f"shift={row['percentage_point_shift']:+.2f} pp"
        )

    lines.append("")

    lines.append("DROPPED FEATURES — R2L / U2R")
    lines.append("-" * 55)

    if dropped_analysis.empty:
        lines.append("No dropped-feature results.")
    else:
        for attack_class in ["R2L", "U2R"]:
            lines.append("")
            lines.append(f"[{attack_class}]")

            subset = dropped_analysis[
                dropped_analysis["class"] == attack_class
            ]

            for _, row in subset.iterrows():
                lines.append(
                    f"{row['feature']}: "
                    f"mean {row['train_mean']:.6g}"
                    f" -> {row['test_mean']:.6g}; "
                    f"median {row['train_median']:.6g}"
                    f" -> {row['test_median']:.6g}; "
                    f"range train "
                    f"[{row['train_min']:.6g}, "
                    f"{row['train_max']:.6g}], "
                    f"test "
                    f"[{row['test_min']:.6g}, "
                    f"{row['test_max']:.6g}]"
                )

    lines.append("")

    lines.append("CATEGORICAL FEATURE SHIFTS")
    lines.append("-" * 55)

    major = categorical_analysis[
        categorical_analysis["shift_type"].isin(
            ["TEST_ONLY", "TRAIN_ONLY", "MAJOR_SHIFT"]
        )
    ]

    if major.empty:
        lines.append("No major categorical shifts detected.")
    else:
        for attack_class in ["R2L", "U2R"]:
            lines.append("")
            lines.append(f"[{attack_class}]")

            subset = major[
                major["class"] == attack_class
            ].sort_values(
                "percentage_point_shift",
                key=lambda s: s.abs(),
                ascending=False,
            )

            for _, row in subset.iterrows():
                lines.append(
                    f"{row['feature']}={row['category']}: "
                    f"{row['train_proportion'] * 100:.2f}% "
                    f"-> "
                    f"{row['test_proportion'] * 100:.2f}% "
                    f"({row['percentage_point_shift']:+.2f} pp) "
                    f"[{row['shift_type']}]"
                )

    lines.append("")

    lines.append("SERVICE SHIFT")
    lines.append("-" * 55)

    for attack_class in ["R2L", "U2R"]:
        lines.append("")
        lines.append(f"[{attack_class}]")

        subset = service_analysis[
            service_analysis["class"] == attack_class
        ].copy()

        subset["abs_shift"] = subset[
            "percentage_point_shift"
        ].abs()

        subset = subset.sort_values(
            "abs_shift",
            ascending=False,
        ).head(15)

        for _, row in subset.iterrows():
            lines.append(
                f"{row['service']}: "
                f"{row['train_proportion'] * 100:.2f}% "
                f"-> "
                f"{row['test_proportion'] * 100:.2f}% "
                f"({row['percentage_point_shift']:+.2f} pp)"
            )

    lines.append("")

    lines.append("LOG1P TRANSFORMATION ANALYSIS")
    lines.append("-" * 55)

    for _, row in transformation_analysis.iterrows():
        lines.append(
            f"{row['class']} / {row['feature']}: "
            f"raw mean "
            f"{row['raw_train_mean']:.6g}"
            f" -> "
            f"{row['raw_test_mean']:.6g}; "
            f"log1p mean "
            f"{row['log1p_train_mean']:.6g}"
            f" -> "
            f"{row['log1p_test_mean']:.6g}; "
            f"raw median "
            f"{row['raw_train_median']:.6g}"
            f" -> "
            f"{row['raw_test_median']:.6g}; "
            f"log1p median "
            f"{row['log1p_train_median']:.6g}"
            f" -> "
            f"{row['log1p_test_median']:.6g}"
        )

    lines.append("")

    lines.append("NUMERICAL FEATURE SHIFTS")
    lines.append("-" * 55)

    for attack_class in ["R2L", "U2R"]:
        lines.append("")
        lines.append(f"[{attack_class}]")

        subset = numeric_analysis[
            numeric_analysis["class"] == attack_class
        ].copy()

        subset["abs_mean_difference"] = subset[
            "mean_difference"
        ].abs()

        subset = subset.sort_values(
            "abs_mean_difference",
            ascending=False,
        ).head(15)

        for _, row in subset.iterrows():
            lines.append(
                f"{row['feature']}: "
                f"mean "
                f"{row['train_mean']:.6g}"
                f" -> "
                f"{row['test_mean']:.6g}; "
                f"median "
                f"{row['train_median']:.6g}"
                f" -> "
                f"{row['test_median']:.6g}"
            )

    lines.append("")

    lines.append("KEY OBSERVATIONS")
    lines.append("-" * 55)
    lines.append(
        "1. This report describes distributional evidence only; "
        "it does not establish causal feature importance."
    )
    lines.append(
        "2. A train/test distribution shift does not by itself "
        "prove that a feature should be changed or restored."
    )
    lines.append(
        "3. Dropped features are reported separately so their "
        "potential diagnostic value can be reviewed."
    )
    lines.append(
        "4. R2L and U2R are analyzed separately because their "
        "sample sizes and attack composition differ substantially."
    )
    lines.append(
        "5. Any future feature-policy change must be evaluated "
        "using training-side development data rather than repeatedly "
        "optimizing against KDDTest+."
    )
    lines.append("")

    lines.append(
        "END OF DIAGNOSTIC FEATURE FAILURE ANALYSIS"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path

# Main
def main():
    print("=" * 60)
    print("NETRA — Feature-Level Failure Analysis")
    print("=" * 60)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Feature config not found: {CONFIG_PATH}"
        )

    print("\nLoading configuration...")
    config = load_config()

    print("Loading KDDTrain+...")
    train = load_dataset(TRAIN_PATH)

    print("Loading KDDTest+...")
    test = load_dataset(TEST_PATH)

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")

    print("\nAnalyzing attack distribution...")
    attack_distribution = analyze_attack_distribution(
        train,
        test,
    )

    print("Analyzing numerical features...")
    numeric_analysis = analyze_numeric_features(
        train,
        test,
    )

    print("Analyzing dropped features...")
    dropped_analysis = analyze_dropped_features(
        train,
        test,
        config["features"]["drop"],
    )

    print("Analyzing categorical features...")
    categorical_analysis = analyze_categorical_features(
        train,
        test,
    )

    print("Analyzing log1p transformations...")
    transformation_analysis = analyze_transformations(
        train,
        test,
        TRANSFORM_FEATURES,
    )

    print("Analyzing service distributions...")
    service_analysis = analyze_service_shift(
        train,
        test,
    )

    # -------------------------------------------------------------
    # Save CSV artifacts
    # -------------------------------------------------------------

    attack_distribution.to_csv(
        OUTPUT_DIR / "attack_distribution.csv",
        index=False,
    )

    numeric_analysis.to_csv(
        OUTPUT_DIR / "numeric_feature_analysis.csv",
        index=False,
    )

    dropped_analysis.to_csv(
        OUTPUT_DIR / "dropped_features_analysis.csv",
        index=False,
    )

    categorical_analysis.to_csv(
        OUTPUT_DIR / "categorical_feature_analysis.csv",
        index=False,
    )

    transformation_analysis.to_csv(
        OUTPUT_DIR / "transformation_analysis.csv",
        index=False,
    )

    service_analysis.to_csv(
        OUTPUT_DIR / "service_shift_analysis.csv",
        index=False,
    )

    print("\nWriting report...")

    report_path = write_report(
        train=train,
        test=test,
        attack_distribution=attack_distribution,
        numeric_analysis=numeric_analysis,
        dropped_analysis=dropped_analysis,
        categorical_analysis=categorical_analysis,
        transformation_analysis=transformation_analysis,
        service_analysis=service_analysis,
        config=config,
    )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"\nOutput directory:")
    print(f"  {OUTPUT_DIR}")

    print("\nGenerated files:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"  - {path.name}")

    print("\nAttack distribution:")
    print(
        attack_distribution.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 60)
    print("IMPORTANT")
    print("=" * 60)
    print(
        "Do not modify the feature policy based solely on "
        "KDDTest+ observations."
    )
    print(
        "Use this report to identify hypotheses for subsequent "
        "training-side experiments."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise