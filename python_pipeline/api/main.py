from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException

from python_pipeline.api.schemas import (
    NetworkTraffic,
    PredictionResponse,
    SHAPFeature,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SERVING_DIR = (
    PROJECT_ROOT
    / "python_pipeline"
    / "models"
    / "serving"
)


MODEL_PATH = (
    SERVING_DIR
    / "model.pkl"
)

PREPROCESSOR_PATH = (
    SERVING_DIR
    / "preprocessor.pkl"
)


CLASS_NAMES = [
    "normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R",
]


RAW_FEATURE_NAMES = [
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


app = FastAPI(
    title="NETRA",
    description=(
        "Network Threat Recognition & Analysis "
        "intrusion detection API"
    ),
    version="1.0.0",
)


model = None
preprocessor = None
explainer = None


@app.on_event("startup")
def load_artifacts():
    global model
    global preprocessor
    global explainer

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(
            "Preprocessor artifact not found: "
            f"{PREPROCESSOR_PATH}"
        )

    with open(
        MODEL_PATH,
        "rb",
    ) as f:
        model = pickle.load(f)

    with open(
        PREPROCESSOR_PATH,
        "rb",
    ) as f:
        preprocessor = pickle.load(f)

    explainer = shap.TreeExplainer(
        model
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded":
            preprocessor is not None,
        "explainer_loaded":
            explainer is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    traffic: NetworkTraffic,
):
    if (
        model is None
        or preprocessor is None
        or explainer is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    # Convert Pydantic model into one-row DataFrame.
    data = traffic.model_dump()

    row = pd.DataFrame(
        [data],
        columns=RAW_FEATURE_NAMES,
    )

    try:
        processed = (
            preprocessor.transform(row)
        )

        probabilities = (
            model.predict_proba(
                processed
            )[0]
        )

        predicted_id = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASS_NAMES[
            predicted_id
        ]

        confidence = float(
            probabilities[predicted_id]
        )

        shap_values = explainer.shap_values(
            processed
        )

        if isinstance(
            shap_values,
            list,
        ):
            class_values = np.asarray(
                shap_values[
                    predicted_id
                ][0]
            )
        else:
            values = np.asarray(
                shap_values
            )

            if values.ndim == 3:
                if (
                    values.shape[1]
                    == processed.shape[1]
                ):
                    class_values = values[
                        0,
                        :,
                        predicted_id,
                    ]
                else:
                    class_values = values[
                        0,
                        predicted_id,
                        :,
                    ]
            else:
                raise ValueError(
                    "Unexpected SHAP output shape."
                )

        feature_names = (
            processed.columns.tolist()
        )

        top_indices = np.argsort(
            np.abs(class_values)
        )[::-1][:3]

        top_features = []

        for index in top_indices:
            value = float(
                class_values[index]
            )

            top_features.append(
                SHAPFeature(
                    feature=feature_names[
                        index
                    ],
                    shap_value=value,
                    absolute_shap_value=abs(
                        value
                    ),
                )
            )

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            top_features=top_features,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )