"""
app.py

ChennaiTransit AI — Passenger Demand Prediction API.

Wraps the existing PredictionPipeline with proper startup handling,
input validation, structured errors, and logging.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, status

from ChennaiTransit_AI.app.schemas import PredictionInput, PredictionResponse
from ChennaiTransit_AI.pipelines.prediction_pipeline import PredictionPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# If the project already has a shared logging_setup/logger.py, import
# get_logger() from there instead of calling basicConfig() here, to avoid
# configuring the root logger twice.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("chennai_transit_ai")

# ---------------------------------------------------------------------------
# Model holder
# ---------------------------------------------------------------------------
ml_models: Dict[str, PredictionPipeline] = {}

MODEL_NAME = "ChennaiTransit_PredictionPipeline_v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads PredictionPipeline exactly once at process startup, instead of
    at import time (as `prediction_pipeline = PredictionPipeline()` at
    module level did originally). This keeps model loading inside a
    controlled lifecycle event, so /health can report whether it actually
    succeeded, and startup failures don't crash module import.
    """
    logger.info("Starting ChennaiTransit AI API — loading prediction pipeline...")
    try:
        ml_models["pipeline"] = PredictionPipeline()
        logger.info("Prediction pipeline loaded successfully.")
    except Exception as exc:
        logger.exception("Failed to load prediction pipeline: %s", exc)
        ml_models["pipeline"] = None

    yield  # ---- app runs here ----

    logger.info("Shutting down ChennaiTransit AI API.")
    ml_models.clear()


app = FastAPI(
    title="Chennai Transit AI",
    description="Passenger Demand Prediction API",
    version="1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------
@app.get(
    "/",
    tags=["General"],
    summary="API status overview",
    description="Returns basic information about the API: name, version, and status.",
)
def home() -> dict:
    return {
        "api_name": "Chennai Transit AI — Passenger Demand Prediction",
        "version": app.version,
        "status": "running",
        "message": "Chennai Transit AI API is running",
    }


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    tags=["General"],
    summary="Health check",
    description="Reports whether the API is up and whether the prediction model is loaded.",
)
def health_check() -> dict:
    model_loaded = ml_models.get("pipeline") is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict passenger demand",
    description=(
        "Accepts station, location, time, weather, and recent passenger-history "
        "features, and returns the predicted passenger count."
    ),
)
def predict_demand(input_data: PredictionInput) -> PredictionResponse:
    pipeline = ml_models.get("pipeline")

    if pipeline is None:
        logger.error("Prediction requested but pipeline is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not available right now. Please try again later.",
        )

    payload = input_data.model_dump()
    logger.info("Received prediction request: %s", payload)

    try:
        raw_prediction = pipeline.predict(payload)
    except Exception as exc:
        # Keeps pipeline-internal errors (bad feature shape, missing
        # artifact, etc.) out of the client response while preserving
        # the full traceback in the server logs.
        logger.exception("Prediction failed for input %s: %s", payload, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error. Please try again.",
        ) from exc

    try:
        predicted_value = round(float(raw_prediction), 2)
    except (TypeError, ValueError) as exc:
        logger.exception("Pipeline returned a non-numeric prediction: %s", raw_prediction)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction result was invalid. Please try again.",
        ) from exc

    logger.info("Prediction successful: %s passengers", predicted_value)

    return PredictionResponse(
        predicted_passenger_count=predicted_value,
        model_name=MODEL_NAME,
        status="success",
    )