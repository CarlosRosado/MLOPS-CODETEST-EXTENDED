from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, Field, ValidationError, field_validator
from typing import List, Dict
import joblib
import yaml
from prometheus_client import start_http_server, Summary, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi.openapi.utils import get_openapi
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.pytorch_classifier import PytorchClassifier
from models.sklearn_classifier import SklearnClassifier
from utils import load_labels, format_response

app = FastAPI()

# Create a metric to track time spent and requests made.
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests')

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
sklearn_model_path = os.path.join(BASE_DIR, 'models', 'sklearn.model')
pytorch_model_path = os.path.join(BASE_DIR, 'models', 'pytorch.model')

sklearn_model = SklearnClassifier(sklearn_model_path)
pytorch_model = PytorchClassifier(pytorch_model_path)

# Load labels
labels = load_labels(os.path.join(BASE_DIR, 'models', 'output_labels.txt'))

class CrystalData(BaseModel):
    crystalData: List[conlist(float, min_length=4, max_length=4)]

class AstromechData(BaseModel):
    crystalData: List[conlist(float, min_length=4, max_length=4)]
    model: str

class PredictionScores(BaseModel):
    blue: float
    green: float
    yellow: float

class PredictionResponse(BaseModel):
    prediction: List[str]
    scores: List[PredictionScores]

@app.post("/sklearn", response_model=PredictionResponse)
@REQUEST_TIME.time()
@REQUEST_COUNT.count_exceptions()
def sklearn_endpoint(data: CrystalData):
    """
    Endpoint for making predictions using the Scikit-Learn model.

    Expects a JSON payload with the key 'crystalData' containing a list of samples.

    Args:
        data (CrystalData): The input data containing a list of samples.

    Returns:
        PredictionResponse: JSON response with the prediction and scores for each label.
    """
    try:
        predictions = sklearn_model.predict(data.crystalData)
        response = format_response(predictions, labels)
        return PredictionResponse(prediction=response["prediction"], scores=response["scores"])
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pytorch", response_model=PredictionResponse)
@REQUEST_TIME.time()
@REQUEST_COUNT.count_exceptions()
def pytorch_endpoint(data: CrystalData):
    """
    Endpoint for making predictions using the PyTorch model.

    Expects a JSON payload with the key 'crystalData' containing a list of samples.

    Args:
        data (CrystalData): The input data containing a list of samples.

    Returns:
        PredictionResponse: JSON response with the prediction and scores for each label.
    """
    try:
        predictions = pytorch_model.predict(data.crystalData)
        response = format_response(predictions, labels)
        return PredictionResponse(prediction=response["prediction"], scores=response["scores"])
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/astromech", response_model=PredictionResponse)
@REQUEST_TIME.time()
@REQUEST_COUNT.count_exceptions()
def astromech_endpoint(data: Dict):
    """
    Endpoint for making predictions using either the Scikit-Learn or PyTorch model.

    Expects a JSON payload with the key 'crystalData' containing a list of samples,
    and a key 'model' specifying either 'sklearn' or 'pytorch'.

    Args:
        data (Dict): The input data containing a list of samples and the model type.

    Returns:
        PredictionResponse: JSON response with the prediction and scores for each label.
    """
    # Check model validity first
    if data.get('model') not in ['sklearn', 'pytorch']:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    try:
        # Validate the rest of the data
        validated_data = AstromechData(**data)

        if validated_data.model == 'sklearn':
            predictions = sklearn_model.predict(validated_data.crystalData)
        else:
            predictions = pytorch_model.predict(validated_data.crystalData)
        response = format_response(predictions, labels)
        return PredictionResponse(prediction=response["prediction"], scores=response["scores"])
    
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    """
    Root endpoint.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the SeedTag Text Classifier API"}

# Load OpenAPI specification from YAML file
with open("src/prediction-openapi.yaml", "r") as f:
    openapi_spec = yaml.safe_load(f)

@app.get("/specifications")
def get_specifications():
    """
    Endpoint to get the OpenAPI specifications.

    Returns:
        dict: The OpenAPI specifications.
    """
    return openapi_spec

@app.get("/metrics")
def metrics():
    """
    Endpoint to get the Prometheus metrics.

    Returns:
        Response: The Prometheus metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Start up the server to expose the metrics.
start_http_server(9090)

def custom_openapi():
    """
    Custom OpenAPI schema.

    Returns:
        dict: The OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema
    with open("src/prediction-openapi.yaml", "r") as f:
        openapi_spec = yaml.safe_load(f)
    openapi_schema = get_openapi(
        title=openapi_spec["info"]["title"],
        version=openapi_spec["info"]["version"],
        description=openapi_spec["info"]["description"],
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=3000)