from pathlib import Path

import numpy as np
import pandas as pd


# Paths
ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "KDDTrain+.txt"
TEST_PATH = ROOT / "data" / "raw" / "KDDTest+.txt"

OUTPUT_DIR = (
    ROOT
    / "python_pipeline"
    / "models"
    / "failure_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



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


CLASS_NAMES = [
    "normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R",
]

# Attack → 5-class mapping
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

# Analysis configuration
CATEGORICAL_FEATURES = [
    "protocol_type",
    "service",
    "flag",
]

NUMERICAL_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_same_srv_rate",
]


FOCUS_CLASSES = [
    "R2L",
    "U2R",
]

# Data loading
def load_dataset(path):
    """
    Load an NSL-KDD dataset and create the five-class target.
    """

    df = pd.read_csv(
        path,
        header=None,
        names=FEATURE_NAMES + [
            "label",
            "difficulty",
        ],
    )

    df["attack_class"] = (
        df["label"]
        .map(ATTACK_CLASS_MAP)
    )

    unknown_labels = sorted(
        df.loc[
            df["attack_class"].isna(),
            "label",
        ]
        .astype(str)
        .unique()
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown attack labels in "
            f"{path.name}: {unknown_labels}"
        )

    return df

# Dataset overview
def build_dataset_overview(train_df, test_df):
    rows = []

    for dataset_name, df in [
        ("KDDTrain+", train_df),
        ("KDDTest+", test_df),
    ]:
        rows.append(
            {
                "dataset": dataset_name,
                "rows": len(df),
                "columns": len(df.columns),
            }
        )

    return pd.DataFrame(rows)


def build_class_distribution(train_df, test_df):
    rows = []

    for attack_class in CLASS_NAMES:

        train_count = int(
            (
                train_df["attack_class"]
                == attack_class
            ).sum()
        )

        test_count = int(
            (
                test_df["attack_class"]
                == attack_class
            ).sum()
        )

        rows.append(
            {
                "attack_class": attack_class,
                "train_count": train_count,
                "test_count": test_count,
                "train_percentage": (
                    train_count
                    / len(train_df)
                    * 100
                ),
                "test_percentage": (
                    test_count
                    / len(test_df)
                    * 100
                ),
            }
        )

    return pd.DataFrame(rows)


# Attack subtype distribution
def build_attack_distribution(train_df, test_df):

    train_counts = (
        train_df["label"]
        .value_counts()
    )

    test_counts = (
        test_df["label"]
        .value_counts()
    )

    all_labels = sorted(
        set(train_counts.index)
        | set(test_counts.index)
    )

    rows = []

    for label in all_labels:

        train_count = int(
            train_counts.get(label, 0)
        )

        test_count = int(
            test_counts.get(label, 0)
        )

        rows.append(
            {
                "attack_label": label,
                "attack_class": ATTACK_CLASS_MAP[label],
                "train_count": train_count,
                "test_count": test_count,
                "train_present": train_count > 0,
                "test_present": test_count > 0,
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "attack_class",
            "attack_label",
        ]
    ).reset_index(drop=True)


# R2L / U2R subtype analysis
def build_focus_subtype_analysis(
    train_df,
    test_df,
    attack_class,
):
    train_focus = train_df[
        train_df["attack_class"]
        == attack_class
    ]

    test_focus = test_df[
        test_df["attack_class"]
        == attack_class
    ]

    subtypes = sorted(
        set(train_focus["label"].unique())
        | set(test_focus["label"].unique())
    )

    total_train = len(train_focus)
    total_test = len(test_focus)

    rows = []

    for subtype in subtypes:

        train_count = int(
            (
                train_focus["label"]
                == subtype
            ).sum()
        )

        test_count = int(
            (
                test_focus["label"]
                == subtype
            ).sum()
        )

        train_percentage = (
            train_count
            / total_train
            * 100
            if total_train > 0
            else 0.0
        )

        test_percentage = (
            test_count
            / total_test
            * 100
            if total_test > 0
            else 0.0
        )

        rows.append(
            {
                "attack_class": attack_class,
                "subtype": subtype,
                "train_count": train_count,
                "test_count": test_count,
                "train_percentage": train_percentage,
                "test_percentage": test_percentage,
            }
        )

    return pd.DataFrame(rows)


# Train-only / test-only attack labels
def find_unique_subtypes(
    train_df,
    test_df,
):
    train_labels = set(
        train_df["label"].unique()
    )

    test_labels = set(
        test_df["label"].unique()
    )

    train_only = sorted(
        train_labels - test_labels
    )

    test_only = sorted(
        test_labels - train_labels
    )

    return train_only, test_only

# Categorical distribution shift
def build_categorical_shift(
    train_df,
    test_df,
):
    rows = []

    for attack_class in FOCUS_CLASSES:

        train_focus = train_df[
            train_df["attack_class"]
            == attack_class
        ]

        test_focus = test_df[
            test_df["attack_class"]
            == attack_class
        ]

        for feature in CATEGORICAL_FEATURES:

            categories = sorted(
                set(
                    train_focus[feature]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                |
                set(
                    test_focus[feature]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )

            for category in categories:

                train_count = int(
                    (
                        train_focus[feature]
                        .astype(str)
                        == category
                    ).sum()
                )

                test_count = int(
                    (
                        test_focus[feature]
                        .astype(str)
                        == category
                    ).sum()
                )

                train_total = len(
                    train_focus
                )

                test_total = len(
                    test_focus
                )

                train_percentage = (
                    train_count
                    / train_total
                    * 100
                    if train_total > 0
                    else 0.0
                )

                test_percentage = (
                    test_count
                    / test_total
                    * 100
                    if test_total > 0
                    else 0.0
                )

                rows.append(
                    {
                        "attack_class": attack_class,
                        "feature": feature,
                        "category": category,
                        "train_count": train_count,
                        "test_count": test_count,
                        "train_percentage": train_percentage,
                        "test_percentage": test_percentage,
                        "percentage_point_shift": (
                            test_percentage
                            - train_percentage
                        ),
                    }
                )

    return pd.DataFrame(rows)

# Numerical distribution shift
def build_numeric_shift(
    train_df,
    test_df,
):
    rows = []

    for attack_class in FOCUS_CLASSES:

        train_focus = train_df[
            train_df["attack_class"]
            == attack_class
        ]

        test_focus = test_df[
            test_df["attack_class"]
            == attack_class
        ]

        for feature in NUMERICAL_FEATURES:

            train_values = pd.to_numeric(
                train_focus[feature],
                errors="coerce",
            )

            test_values = pd.to_numeric(
                test_focus[feature],
                errors="coerce",
            )

            rows.append(
                {
                    "attack_class": attack_class,
                    "feature": feature,

                    "train_mean": train_values.mean(),
                    "test_mean": test_values.mean(),

                    "train_median": train_values.median(),
                    "test_median": test_values.median(),

                    "train_std": train_values.std(),
                    "test_std": test_values.std(),

                    "train_min": train_values.min(),
                    "test_min": test_values.min(),

                    "train_max": train_values.max(),
                    "test_max": test_values.max(),
                }
            )

    return pd.DataFrame(rows)

# Report helpers
def format_subtype_table(df):

    if df.empty:
        return "No subtypes found.\n"

    output = df.copy()

    numeric_columns = [
        "train_percentage",
        "test_percentage",
    ]

    for column in numeric_columns:
        output[column] = output[column].map(
            lambda value: f"{value:.2f}%"
        )

    return output.to_string(
        index=False
    )


def format_distribution_table(df):

    output = df.copy()

    output["train_percentage"] = (
        output["train_percentage"]
        .map(lambda x: f"{x:.2f}%")
    )

    output["test_percentage"] = (
        output["test_percentage"]
        .map(lambda x: f"{x:.2f}%")
    )

    return output.to_string(
        index=False
    )

# Human-readable report
def write_report(
    train_df,
    test_df,
    class_distribution,
    attack_distribution,
    r2l_analysis,
    u2r_analysis,
    categorical_shift,
    numeric_shift,
    train_only,
    test_only,
):
    report_path = (
        OUTPUT_DIR
        / "failure_analysis_report.txt"
    )

    with open(
        report_path,
        "w",
    ) as f:

        f.write("=" * 70 + "\n")
        f.write(
            "NETRA — FAILURE ANALYSIS\n"
        )
        f.write("=" * 70 + "\n\n")

        # ----------------------------------------------------
        # Dataset overview
        # ----------------------------------------------------

        f.write(
            "1. DATASET OVERVIEW\n"
        )
        f.write("-" * 70 + "\n")

        f.write(
            f"KDDTrain+ rows: {len(train_df)}\n"
        )

        f.write(
            f"KDDTest+ rows:  {len(test_df)}\n"
        )

        f.write(
            f"Train columns:   {len(train_df.columns)}\n"
        )

        f.write(
            f"Test columns:    {len(test_df.columns)}\n\n"
        )

        f.write(
            "Five-class distribution:\n\n"
        )

        f.write(
            format_distribution_table(
                class_distribution
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # Attack subtype distribution
        # ----------------------------------------------------

        f.write(
            "2. ATTACK SUBTYPE DISTRIBUTION\n"
        )
        f.write("-" * 70 + "\n\n")

        f.write(
            attack_distribution.to_string(
                index=False
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # R2L
        # ----------------------------------------------------

        f.write(
            "3. R2L ANALYSIS\n"
        )
        f.write("-" * 70 + "\n\n")

        f.write(
            format_subtype_table(
                r2l_analysis
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # U2R
        # ----------------------------------------------------

        f.write(
            "4. U2R ANALYSIS\n"
        )
        f.write("-" * 70 + "\n\n")

        f.write(
            format_subtype_table(
                u2r_analysis
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # Train-only / test-only
        # ----------------------------------------------------

        f.write(
            "5. TRAIN-ONLY / TEST-ONLY ATTACK SUBTYPES\n"
        )
        f.write("-" * 70 + "\n\n")

        f.write(
            "Present in KDDTrain+ but absent "
            "from KDDTest+:\n"
        )

        if train_only:
            for label in train_only:
                f.write(f"  - {label}\n")
        else:
            f.write(
                "  None\n"
            )

        f.write("\n")

        f.write(
            "Present in KDDTest+ but absent "
            "from KDDTrain+:\n"
        )

        if test_only:
            for label in test_only:
                f.write(f"  - {label}\n")
        else:
            f.write(
                "  None\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        f.write(
            "6. CATEGORICAL DISTRIBUTION SHIFT\n"
        )
        f.write("-" * 70 + "\n\n")

        f.write(
            "Focus classes: R2L and U2R\n"
        )

        f.write(
            "Features: protocol_type, service, flag\n\n"
        )

        # Show largest shifts first
        categorical_display = (
            categorical_shift
            .copy()
            .sort_values(
                "percentage_point_shift",
                key=lambda s: s.abs(),
                ascending=False,
            )
            .head(40)
        )

        f.write(
            categorical_display.to_string(
                index=False
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # Numerical
        # ----------------------------------------------------

        f.write(
            "7. NUMERICAL DISTRIBUTION SHIFT\n"
        )
        f.write("-" * 70 + "\n\n")

        f.write(
            "Focus classes: R2L and U2R\n\n"
        )

        f.write(
            numeric_shift.to_string(
                index=False
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # Key observations
        # ----------------------------------------------------

        f.write(
            "8. KEY OBSERVATIONS\n"
        )
        f.write("-" * 70 + "\n\n")

        # R2L distribution
        r2l_train_total = int(
            r2l_analysis["train_count"].sum()
        )

        r2l_test_total = int(
            r2l_analysis["test_count"].sum()
        )

        u2r_train_total = int(
            u2r_analysis["train_count"].sum()
        )

        u2r_test_total = int(
            u2r_analysis["test_count"].sum()
        )

        f.write(
            f"R2L samples — train: "
            f"{r2l_train_total}, "
            f"test: {r2l_test_total}\n"
        )

        f.write(
            f"U2R samples — train: "
            f"{u2r_train_total}, "
            f"test: {u2r_test_total}\n\n"
        )

        if test_only:
            f.write(
                "Test-only attack subtypes were detected.\n"
            )
        else:
            f.write(
                "No test-only attack subtypes were detected.\n"
            )

        if train_only:
            f.write(
                "Train-only attack subtypes were detected.\n"
            )
        else:
            f.write(
                "No train-only attack subtypes were detected.\n"
            )

        f.write("\n")

        f.write(
            "Largest categorical shifts among "
            "R2L/U2R:\n"
        )

        largest_categorical = (
            categorical_shift
            .copy()
            .assign(
                absolute_shift=lambda x:
                    x["percentage_point_shift"].abs()
            )
            .sort_values(
                "absolute_shift",
                ascending=False,
            )
            .head(10)
        )

        for _, row in largest_categorical.iterrows():
            f.write(
                f"  - {row['attack_class']} / "
                f"{row['feature']} / "
                f"{row['category']}: "
                f"{row['percentage_point_shift']:.2f} "
                f"percentage points\n"
            )

        f.write("\n")

        f.write(
            "Numerical distribution comparisons are "
            "provided in numeric_shift.csv.\n"
        )

        f.write(
            "This report is diagnostic only and does "
            "not prescribe model or feature changes.\n"
        )

    return report_path


# Main
def main():

    print("=" * 70)
    print("NETRA — FAILURE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("Loading KDDTrain+...")

    train_df = load_dataset(
        TRAIN_PATH
    )

    print(
        f"KDDTrain+ rows: {len(train_df)}"
    )

    print()
    print("Loading KDDTest+...")

    test_df = load_dataset(
        TEST_PATH
    )

    print(
        f"KDDTest+ rows: {len(test_df)}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if len(train_df) != 125973:
        raise RuntimeError(
            "Unexpected KDDTrain+ row count: "
            f"{len(train_df)}"
        )

    if len(test_df) != 22544:
        raise RuntimeError(
            "Unexpected KDDTest+ row count: "
            f"{len(test_df)}"
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("Five-class distribution:")

    class_distribution = (
        build_class_distribution(
            train_df,
            test_df,
        )
    )

    print(
        class_distribution.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Attack distribution
    # --------------------------------------------------------

    print()
    print(
        "Building attack subtype distribution..."
    )

    attack_distribution = (
        build_attack_distribution(
            train_df,
            test_df,
        )
    )

    # --------------------------------------------------------
    # R2L / U2R
    # --------------------------------------------------------

    print(
        "Analyzing R2L subtypes..."
    )

    r2l_analysis = (
        build_focus_subtype_analysis(
            train_df,
            test_df,
            "R2L",
        )
    )

    print(
        "Analyzing U2R subtypes..."
    )

    u2r_analysis = (
        build_focus_subtype_analysis(
            train_df,
            test_df,
            "U2R",
        )
    )

    # --------------------------------------------------------
    # Train/test-only labels
    # --------------------------------------------------------

    train_only, test_only = (
        find_unique_subtypes(
            train_df,
            test_df,
        )
    )

    # --------------------------------------------------------
    # Categorical shift
    # --------------------------------------------------------

    print(
        "Analyzing categorical distribution shift..."
    )

    categorical_shift = (
        build_categorical_shift(
            train_df,
            test_df,
        )
    )

    # --------------------------------------------------------
    # Numerical shift
    # --------------------------------------------------------

    print(
        "Analyzing numerical distribution shift..."
    )

    numeric_shift = (
        build_numeric_shift(
            train_df,
            test_df,
        )
    )

    # --------------------------------------------------------
    # Save CSV artifacts
    # --------------------------------------------------------

    print()
    print("Saving artifacts...")

    attack_distribution.to_csv(
        OUTPUT_DIR
        / "attack_distribution.csv",
        index=False,
    )

    r2l_analysis.to_csv(
        OUTPUT_DIR
        / "r2l_subtype_analysis.csv",
        index=False,
    )

    u2r_analysis.to_csv(
        OUTPUT_DIR
        / "u2r_subtype_analysis.csv",
        index=False,
    )

    categorical_shift.to_csv(
        OUTPUT_DIR
        / "categorical_shift.csv",
        index=False,
    )

    numeric_shift.to_csv(
        OUTPUT_DIR
        / "numeric_shift.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    report_path = write_report(
        train_df=train_df,
        test_df=test_df,
        class_distribution=class_distribution,
        attack_distribution=attack_distribution,
        r2l_analysis=r2l_analysis,
        u2r_analysis=u2r_analysis,
        categorical_shift=categorical_shift,
        numeric_shift=numeric_shift,
        train_only=train_only,
        test_only=test_only,
    )

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FAILURE ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Train-only subtypes: "
        f"{len(train_only)}"
    )

    print(
        f"Test-only subtypes:  "
        f"{len(test_only)}"
    )

    print()
    print("Artifacts:")

    print(
        f"  Attack distribution: "
        f"{OUTPUT_DIR / 'attack_distribution.csv'}"
    )

    print(
        f"  R2L analysis:        "
        f"{OUTPUT_DIR / 'r2l_subtype_analysis.csv'}"
    )

    print(
        f"  U2R analysis:        "
        f"{OUTPUT_DIR / 'u2r_subtype_analysis.csv'}"
    )

    print(
        f"  Categorical shift:   "
        f"{OUTPUT_DIR / 'categorical_shift.csv'}"
    )

    print(
        f"  Numerical shift:     "
        f"{OUTPUT_DIR / 'numeric_shift.csv'}"
    )

    print(
        f"  Report:              "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()