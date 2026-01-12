import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from catboost import CatBoostClassifier

MODEL_PATH = os.path.expanduser(os.getenv("MODEL_PATH", "~/content/model.cbm"))
DATA_PATH = os.path.expanduser(os.getenv("DATA_PATH", "~/content/post_prepare_data.csv"))

model: Optional[CatBoostClassifier] = None
data: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, data
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    HomeTeam: str = Field(..., min_length=1)
    AwayTeam: str = Field(..., min_length=1)
    season: str = Field(default="2024-2025", min_length=1)
    
    @field_validator('HomeTeam', 'AwayTeam', 'season')
    @classmethod
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()


def get_match_features(home_team: str, away_team: str, season: str) -> Optional[pd.DataFrame]:
    if data is None:
        return None
    mask = (
        (data["HomeTeam_season_1"] == home_team) & 
        (data["AwayTeam_season_1"] == away_team) & 
        (data["Season_season_1"] == season)
    )
    match_data = data[mask]
    if match_data.empty:
        return None
    return match_data.iloc[[0]]


def prepare_features(match_data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ["target", "Season_season_1", "Date_season_1"]
    columns_to_drop = [col for col in columns_to_drop if col in match_data.columns]
    if match_data.columns[0] == "" or match_data.columns[0] == "Unnamed: 0":
        columns_to_drop.append(match_data.columns[0])
    return match_data.drop(columns=columns_to_drop, errors='ignore')


def make_prediction(features: pd.DataFrame) -> dict:
    if model is None:
        raise ValueError("Model not loaded")
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    labels = {0: "home win", 1: "away win", 2: "draw"}
    
    pred_value = int(prediction.flat[0])
    prob_array = probabilities[0]
    
    return {
        "prediction": pred_value,
        "prediction_label": labels.get(pred_value, "unknown"),
        "probabilities": {
            "home_win": float(prob_array[0]),
            "away_win": float(prob_array[1]),
            "draw": float(prob_array[2])
        }
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raise HTTPException(status_code=400, detail="bad request")


@app.post("/forward")
async def forward(request: PredictionRequest):
    try:
        match_data = get_match_features(request.HomeTeam, request.AwayTeam, request.season)
        if match_data is None:
            raise HTTPException(status_code=403, detail="модель не смогла обработать данные")
        
        features = prepare_features(match_data)
        result = make_prediction(features)
        return result
    except Exception:
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")


@app.get("/teams")
async def get_teams():
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    home_teams = sorted(data["HomeTeam_season_1"].unique().tolist())
    away_teams = sorted(data["AwayTeam_season_1"].unique().tolist())
    all_teams = sorted(list(set(home_teams + away_teams)))
    return {"teams": all_teams}


@app.get("/seasons")
async def get_seasons():
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    seasons = sorted(data["Season_season_1"].unique().tolist())
    return {"seasons": seasons}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

