"""FastAPI service exposing the two-stage email classifier."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionResponse, PredictRequest

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load both cascade models once when the service starts."""
    app.state.spam_model = None
    app.state.normal_model = None
    try:
        app.state.spam_model = joblib.load(ARTIFACT_DIR / "spam_model.joblib")
        app.state.normal_model = joblib.load(ARTIFACT_DIR / "normal_model.joblib")
        logger.info("Cascade models loaded from %s", ARTIFACT_DIR)
    except Exception:  # pragma: no cover - only reached when artifacts are missing
        logger.exception("Could not load cascade models from %s", ARTIFACT_DIR)
    yield


app = FastAPI(
    title="Email Cascade Classifier",
    description="Spam detection followed by category prediction for normal emails.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, object]:
    """Report service health and whether both models are loaded in memory."""
    models_loaded = app.state.spam_model is not None and app.state.normal_model is not None
    return {"status": "ok", "models_loaded": models_loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest) -> PredictionResponse:
    """Classify an email: spam first, otherwise one of three normal categories."""
    spam_model = app.state.spam_model
    normal_model = app.state.normal_model
    if spam_model is None or normal_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    if int(spam_model.predict([payload.text])[0]) == 1:
        return PredictionResponse(is_spam=True, category="spam")

    category = str(normal_model.predict([payload.text])[0])
    return PredictionResponse(is_spam=False, category=category)
