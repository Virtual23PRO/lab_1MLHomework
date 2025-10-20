from fastapi import FastAPI
from pydantic import BaseModel, Field
from inference import get_pipeline

app = FastAPI(title="ML API")


class PredictIn(BaseModel):
    text: str = Field(min_length=1, pattern=r"\S+")


class PredictOut(BaseModel):
    prediction: str


@app.on_event("startup")
def _warmup():
    get_pipeline()


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> PredictOut:
    label = get_pipeline().predict(payload.text.strip())
    return PredictOut(prediction=label)
