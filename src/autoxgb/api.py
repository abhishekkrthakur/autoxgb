import os

from fastapi import FastAPI

from .predict import AutoXGBPredict


app = FastAPI()
axgp = AutoXGBPredict(model_path=os.environ.get("AUTOXGB_MODEL_PATH"))
schema = axgp.get_prediction_schema()


@app.post("/predict")
def predict(sample: schema):
    sample = sample.json()
    return axgp.predict_single(sample)
