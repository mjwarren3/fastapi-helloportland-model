from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd
from utils import select_text_data, select_numeric_data

# Load the trained model
model = load('model4.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define a Pydantic model for the input data
class InputData(BaseModel):
    short_description: str
    isNight: int
    containsPortland: int

# Define the predict route
@app.post("/predict")
async def predict(input_data: InputData):
    try:
        def select_text_data(x):
            return x['short_description']

        def select_numeric_data(x):
            return x[['isNight', 'containsPortland']]

        # Convert input data to dataframe
        input_df = pd.DataFrame([dict(input_data)])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Convert prediction to JSON format and return
        return {"predicted_clicks": prediction[0]}
    
    except:
        raise HTTPException(status_code=400, detail="Prediction failed")
