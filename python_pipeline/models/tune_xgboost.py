from pathlib import Path
import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from python_pipeline.preprocessing.preprocessor import NETRApreprocessor

#Paths
ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "KDDTrain+.txt"
CONFIG_PATH = ROOT / "python_pipeline" / "config" / "feature_config.yaml"

OUTPUT_DIR = ROOT / "python_pipeline" / "models" / "optuna_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = OUTPUT_DIR / "xgboost_study.db"
BEST_PARAMS_PATH = OUTPUT_DIR / "best_params.json"
TRIALS_PATH = OUTPUT_DIR / "trials.csv"

#Reproducability
RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 30

#NSL-KDD Schema
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

#Data Loading
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

#Optuna objective
def objective(trial, X, y):
    n_estimators = trial.suggest_int(
        "n_estimators",
        200,
        600,
    )

    max_depth = trial.suggest_int(
        "n_estimators",
        3,
        15,
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        0.02,
        0.20,
        log=True, 
    )

    subsample = trial.suggest_float(
        "subsample",
        0.70,
        1.00,
    )

    colsample_bytree = trial.suggest_float(
        "colsample_bytree",
        0.70,
        1.00,
    )

    weight_multiplier = trial.suggest_float(
        "weight_multiplier",
        0.50,
        2.00,
    )

    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": "multi:softprob",
        "num_class": len(CLASS_NAMES),
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    #Stratified Cross Validation
    y_encoded = y.map(CLASS_TO_ID).to_numpy()

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y_encoded),
        start=1,
    ):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()

        y_train = y_encoded[train_idx]
        y_val = y_encoded[val_idx]

        preprocessor = NETRApreprocessor(CONFIG_PATH)

        X_train_preprocessed = preprocessor.fit_transform(
            X_train
        )
        X_val_preprocessed = preprocessor.transform(
            X_val
        )

        '''
        Balanced sample weights with tunable strength.
        multiplier = 1.0 → normal balanced weighting
        multiplier < 1 → less aggressive
        multiplier > 1 → more aggressive
        '''

        balanced_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        sample_weights = np.ones_like(
            balanced_weights,
            dtype=float,
        )

        minority_mask = y_train >= CLASS_TO_ID["R2L"]

        sample_weights[minority_mask] = (
            1.0
            + weight_multiplier
            + (balanced_weights[minority_mask] - 1.0)
        )

        #Train
        model = XGBClassifier(**model_params)

        model.fit(
            X_train_preprocessed,
            y_train,
            sample_weight=sample_weights,
        )

        predictions = model.predict(
            X_val_preprocessed
        )

        fold_macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )
        fold_scores.append(fold_macro_f1)

        #Reporting immediate results so Optuna can prune
        trial.report(
            float(np.mean(fold_scores)),
            step=fold,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_macro_f1 = float(np.mean(fold_scores))
    return mean_macro_f1

#Main
def main():
    X, y = load_training_data()

    storage = (
        f"sqlite:///{DB_PATH}"
    )

    study = optuna.create_study(
        study_name="netra_xgboost_5class",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        )
    )
    print("\n")
    print("=" * 70)
    print("NETRA — OPTUNA XGBOOST TUNING")
    print("=" * 70)
    print(f"Trials:       {N_TRIALS}")
    print(f"CV folds:     {N_SPLITS}")
    print("Objective:    Macro-F1")
    print("Model:        Weighted XGBoost")
    print("KDDTest+:     NOT USED")
    print("=" * 70)

    start = time.perf_counter()

    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
        ),
        n_trials=N_TRIALS,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    elapsed = time.perf_counter() - start

    #Best Result
    print("\n")
    print("=" * 70)
    print("OPTUNA COMPLETE")
    print("=" * 70)

    print(f"Completed trials: {len(study.trials)}")
    print(f"Best Macro-F1:    {study.best_value:.6f}")
    print(f"Total time:       {elapsed:.3f}s")

    print("\nBest Parameters:")

    for parameter, value in study.best_params.items():
        print(f"  {parameter}: {value}")

    best_results = {
        "best_macro_f1": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "cv_folds": N_SPLITS,
        "random_state": RANDOM_STATE,
    }

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(
            best_results,
            f,
            indent=2,
        )

    #Trial History
    trials_df = study.trials_dataframe()

    trials_df.to_csv(
        TRIALS_PATH,
        index=False,
    )
    print("\nArtifacts:")
    print(f"  Study DB:       {DB_PATH}")
    print(f"  Best params:    {BEST_PARAMS_PATH}")
    print(f"  Trial history:  {TRIALS_PATH}")

    print("\nKDDTest+ remains untouched.")

if __name__ == "__main__":
    main()