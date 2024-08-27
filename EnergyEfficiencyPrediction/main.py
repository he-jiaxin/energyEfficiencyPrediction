from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the models
model_heating = joblib.load("/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_heating_corrected.joblib")
model_cooling = joblib.load("/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_cooling_corrected.joblib")

class PredictionRequest(BaseModel):
    input: list

@app.get("/")
def read_root():
    return {"message": "Welcome to the Energy Efficiency Prediction API!"}

@app.post('/predict_heating')
def predict_heating(request: PredictionRequest):
    input_data = np.array(request.input).reshape(1, -1)
    try:
        prediction = model_heating.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'prediction': prediction.tolist()}

@app.post('/predict_cooling')
def predict_cooling(request: PredictionRequest):
    input_data = np.array(request.input).reshape(1, -1)
    try:
        prediction = model_cooling.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)