
import numpy as np
import os
import pandas as pd
from typing import Dict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import keras



app = FastAPI()



def load_model(model_filename):
    """Load model from a file.

    This function loads model from the specified file.

    Args:
        filename (str): The filename of the keras model.

    Returns:
        model
    """
    
    # Get the directory of the currently executing app
    app_directory = os.path.dirname(os.path.abspath(__file__))

    # # Construct the full path to the model file
    model_path = os.path.join(app_directory, model_filename)

    model = keras.saving.load_model(model_path)

    return model

def prepare_features(pixels):
    """
    Prepare the input features for prediction 

    Parameters:
        pixels (Dict[str, float]): A dictionary containing pixel information.

    Returns:
        pd.DataFrame: A DataFrame with the pixels.
    """
    df = pd.DataFrame([pixels])
    X = df.values / 255.0
    X = X.reshape(-1, 28, 28, 1)

    return X

def predict(X):
    """
    Make a prediction using the input features.

    Parameters:
        features (pd.DataFrame): DataFrame containing the input features.

    Returns:
        int: The predicted outcome (1 to 10).
    """
    model = load_model("digits_model1.h5")

    pred = model.predict(X)

    pred_arg = np.argmax(pred, axis=1)
    
    return int(pred_arg[0])


@app.post("/predict", status_code=200)
def predict_endpoint(pixels: Dict[str, float]):
    """
    Endpoint to make a prediction based on pixels.

    Parameters:
        pixels (Dict[str, float]): A dictionary containing pixels.

    Returns:
        JSONResponse: A JSON response containing the predicted outcome.
    """
    try:
        
        features = prepare_features(pixels)
        
        pred = predict(features)

        result = {
            'digit': pred
        }

        return JSONResponse(content=result)

    except Exception as e:
        # Handle any errors that might occur during processing
        error_msg = {'predict_endpoint error': str(e)}
        return JSONResponse(content=error_msg, status_code=500)

@app.get("/healthcheck")  # Route for ELB health checks
def healthcheck():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)