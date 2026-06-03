import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.dl_predictor import DLPredictor  # noqa: E402

predictor: Optional[DLPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    os.chdir(ROOT_DIR)
    predictor = DLPredictor()
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raise HTTPException(status_code=400, detail="bad request")


@app.post("/forward")
async def forward(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict(request.HomeTeam, request.AwayTeam, request.season)
    except Exception:
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")
    if result is None:
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")
    return result


@app.get("/teams")
async def get_teams():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    return {"teams": predictor.teams()}


@app.get("/seasons")
async def get_seasons():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    return {"seasons": predictor.seasons()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
