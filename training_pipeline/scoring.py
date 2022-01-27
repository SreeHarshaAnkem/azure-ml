
import json
import pickle
import numpy as np
import os

def init():
    global model, scaler
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "wine-quality-lr/1/model.pkl")
    model = pickle.load(model_path)
    
    scaler_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "wine-quality-scaler/15/scaler.pkl")
    scaler = pickle.load(scaler_path)
    
    
def run(raw_data):
    data = np.array(json.loads(raw_data)["data"])
    prepped = scaler.transform(data)
    predictions = model.predict(prepped)
    return predictions.tolist()

    
