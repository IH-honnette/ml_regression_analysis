"""FastAPI service for engineering-graduate salary prediction.

Endpoints
---------
GET  /            health check + metadata
POST /predict     predict salary from 12 validated inputs
POST /retrain     re-fit the model from an uploaded CSV and hot-swap it

Run locally:  uv run uvicorn prediction:app --reload --app-dir summative/API
Swagger UI:   <base-url>/docs
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODELS = Path(__file__).resolve().parent / "models"

FEATURES = [
    "CollegeTier", "collegeGPA", "English", "Logical", "Quant", "Domain",
    "ComputerProgramming", "conscientiousness", "agreeableness",
    "extraversion", "nueroticism", "openess_to_experience",
]

# ---------------------------------------------------------------------------
# Model state (loaded once, replaceable by /retrain)
# ---------------------------------------------------------------------------
class ModelState:
    def __init__(self) -> None:
        self.model = joblib.load(MODELS / "best_model.pkl")
        self.scaler = joblib.load(MODELS / "scaler.pkl")
        self.meta = joblib.load(MODELS / "metadata.pkl")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X[FEATURES]))


state = ModelState()

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Engineering Graduate Salary Prediction API",
    description="Predicts an engineering graduate's starting salary (INR) from "
                "academic, aptitude and personality features.",
    version="1.0.0",
)

# CORS is configured with an explicit allow-list rather than a wildcard.
#   * allow_origins  — only the Flutter app hosts and our own docs may call the
#                      API from a browser; unknown origins are rejected.
#   * allow_methods  — the API only serves GET (health) and POST (predict,
#                      retrain), plus the OPTIONS pre-flight; nothing else.
#   * allow_headers  — limited to the headers the client actually sends.
#   * allow_credentials = False — the API is stateless and uses no cookies or
#                      auth sessions, so credentialed cross-site requests are
#                      unnecessary and are disabled to reduce attack surface.
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:8080",     # Flutter web dev server
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "capacitor://localhost",     # mobile webview origins
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


# ---------------------------------------------------------------------------
# Request / response schemas with type + range constraints
# ---------------------------------------------------------------------------
class SalaryFeatures(BaseModel):
    CollegeTier: int = Field(..., ge=1, le=2,
                             description="College tier (1 = top tier, 2 = other)")
    collegeGPA: float = Field(..., ge=0, le=100,
                              description="College GPA as a percentage (0–100)")
    English: float = Field(..., ge=0, le=900, description="AMCAT English score")
    Logical: float = Field(..., ge=0, le=900, description="AMCAT Logical score")
    Quant: float = Field(..., ge=0, le=900, description="AMCAT Quant score")
    Domain: float = Field(..., ge=0, le=1,
                          description="Normalized domain-knowledge score (0–1)")
    ComputerProgramming: float = Field(..., ge=0, le=900,
                                       description="Computer Programming score")
    conscientiousness: float = Field(..., ge=-8, le=8)
    agreeableness: float = Field(..., ge=-8, le=8)
    extraversion: float = Field(..., ge=-8, le=8)
    nueroticism: float = Field(..., ge=-8, le=8)
    openess_to_experience: float = Field(..., ge=-8, le=8)

    model_config = {
        "json_schema_extra": {
            "example": {
                "CollegeTier": 1, "collegeGPA": 78.5, "English": 620,
                "Logical": 560, "Quant": 640, "Domain": 0.72,
                "ComputerProgramming": 480, "conscientiousness": 0.5,
                "agreeableness": 0.2, "extraversion": -0.3,
                "nueroticism": 0.1, "openess_to_experience": 0.4,
            }
        }
    }


# Approximate INR -> RWF mid-market rate (~July 2026). Kept as a constant so the
# API stays offline-friendly; update as needed or wire to an FX service later.
INR_TO_RWF = 15.28


class PredictionResponse(BaseModel):
    predicted_salary: float = Field(..., description="Predicted annual salary in INR")
    currency: str = "INR"
    predicted_salary_rwf: float = Field(
        ..., description="Predicted annual salary converted to Rwandan Francs (RWF)")
    inr_to_rwf_rate: float = Field(..., description="INR->RWF rate used")
    model: str


class RetrainResponse(BaseModel):
    status: str
    rows_used: int
    test_rmse: float
    test_r2: float
    model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {
        "message": "Engineering Graduate Salary Prediction API",
        "model": state.meta.get("best_model"),
        "features": FEATURES,
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SalaryFeatures) -> PredictionResponse:
    X = pd.DataFrame([payload.model_dump()])
    salary = float(state.predict(X)[0])
    salary = max(salary, 0.0)  # salary cannot be negative
    return PredictionResponse(
        predicted_salary=round(salary, 2),
        predicted_salary_rwf=round(salary * INR_TO_RWF, 2),
        inr_to_rwf_rate=INR_TO_RWF,
        model=state.meta.get("best_model", "unknown"),
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)) -> RetrainResponse:
    """Re-train the model on a newly uploaded CSV and hot-swap it in memory.

    The CSV must contain the 12 feature columns plus a `Salary` target column.
    On success the live model, scaler and metadata are replaced without a restart.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    missing = [c for c in FEATURES + ["Salary"] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"Missing required columns: {missing}")

    # Same cleaning as training: impute -1 sentinels, cap salary outliers.
    for col in ["ComputerProgramming", "Domain"]:
        df[col] = df[col].replace(-1, np.nan)
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=FEATURES + ["Salary"])
    cap = df["Salary"].quantile(0.99)
    df = df[df["Salary"] <= cap]
    if len(df) < 50:
        raise HTTPException(status_code=422,
                            detail="Not enough valid rows to retrain (need >= 50).")

    X, y = df[FEATURES].astype(float), df["Salary"].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    model = LinearRegression().fit(scaler.transform(X_tr), y_tr)
    pred = model.predict(scaler.transform(X_te))
    rmse = float(mean_squared_error(y_te, pred) ** 0.5)
    r2 = float(r2_score(y_te, pred))

    # Persist and hot-swap.
    joblib.dump(model, MODELS / "best_model.pkl")
    joblib.dump(scaler, MODELS / "scaler.pkl")
    meta = {"features": FEATURES, "best_model": "Linear Regression (retrained)"}
    joblib.dump(meta, MODELS / "metadata.pkl")
    state.model, state.scaler, state.meta = model, scaler, meta

    return RetrainResponse(status="retrained", rows_used=len(df),
                           test_rmse=round(rmse, 2), test_r2=round(r2, 4),
                           model=meta["best_model"])
