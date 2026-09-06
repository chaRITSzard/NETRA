from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.preprocessor import NETRApreprocessor

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "raw" / "KDDTrain+.txt"
CONFIG_PATH = ROOT / "python_pipeline" / "config" / "feature_config.yaml"

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

RAW_COLUMNS = FEATURE_NAMES + ["label", "difficulty"]


# --------------------------------------------------
# Load training data
# --------------------------------------------------

print("Loading KDDTrain+...")

df = pd.read_csv(
    TRAIN_PATH,
    header=None,
    names=RAW_COLUMNS
)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

assert len(df) == 125973
assert len(df.columns) == 43


# --------------------------------------------------
# Separate features
# --------------------------------------------------

X = df[FEATURE_NAMES].copy()


# --------------------------------------------------
# Create and fit preprocessor
# --------------------------------------------------

print("\nFitting NETRA preprocessor...")

preprocessor = NETRApreprocessor(CONFIG_PATH)

X_processed = preprocessor.fit_transform(X)


# --------------------------------------------------
# Basic structural checks
# --------------------------------------------------

print("\n=== PREPROCESSOR TEST ===")

print(f"Input features:     {len(X.columns)}")
print(f"Output features:    {X_processed.shape[1]}")
print(f"Output rows:        {X_processed.shape[0]}")

assert X_processed.shape[0] == len(X)


# --------------------------------------------------
# Check dropped features
# --------------------------------------------------

drop_features = preprocessor.drop_features

remaining_input_features = [
    feature for feature in drop_features
    if feature in X_processed.columns
]

print(f"Dropped features remaining: {remaining_input_features}")

assert not remaining_input_features


# --------------------------------------------------
# Check numeric output
# --------------------------------------------------

print(
    "All output numeric: ",
    all(pd.api.types.is_numeric_dtype(dtype)
        for dtype in X_processed.dtypes)
)

assert all(
    pd.api.types.is_numeric_dtype(dtype)
    for dtype in X_processed.dtypes
)


# --------------------------------------------------
# Check NaN / infinity
# --------------------------------------------------

nan_count = X_processed.isna().sum().sum()
inf_count = np.isinf(X_processed.to_numpy()).sum()

print(f"NaN values:         {nan_count}")
print(f"Infinite values:    {inf_count}")

assert nan_count == 0
assert inf_count == 0


# --------------------------------------------------
# Check frequency encoding
# --------------------------------------------------

print("\nFrequency maps:")

for feature, mapping in preprocessor.frequency_maps.items():
    print(f"  {feature}: {len(mapping)} categories")

assert "service" in preprocessor.frequency_maps
assert len(preprocessor.frequency_maps["service"]) > 0


# --------------------------------------------------
# Check expected transformations
# --------------------------------------------------

for feature in ["duration", "src_bytes", "dst_bytes"]:
    assert feature in X_processed.columns

print("log1p features present: duration, src_bytes, dst_bytes")

# service should now be numeric rather than categorical
assert "service" in X_processed.columns
assert pd.api.types.is_numeric_dtype(X_processed["service"])

print("service frequency encoding: OK")


# --------------------------------------------------
# Check reproducibility of transform()
# --------------------------------------------------

X_again = preprocessor.transform(X)

assert list(X_processed.columns) == list(X_again.columns)
assert np.allclose(
    X_processed.to_numpy(),
    X_again.to_numpy()
)

print("Repeated transform consistency: OK")


# --------------------------------------------------
# Final result
# --------------------------------------------------

print("\n==============================")
print("PREPROCESSOR TEST PASSED")
print("==============================")
