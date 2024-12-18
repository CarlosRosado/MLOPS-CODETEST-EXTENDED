import pytest
from fastapi.testclient import TestClient
import sys, os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app import app

client = TestClient(app)

def test_sklearn_endpoint():
    response = client.post("/sklearn", json={"crystalData": [[0.92, 0.12, 0.31, 0.09]]})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "scores" in response.json()

def test_pytorch_endpoint():
    response = client.post("/pytorch", json={"crystalData": [[0.92, 0.12, 0.31, 0.09]]})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "scores" in response.json()

def test_astromech_endpoint_sklearn():
    response = client.post("/astromech", json={"crystalData": [[0.92, 0.12, 0.31, 0.09]], "model": "sklearn"})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "scores" in response.json()

def test_astromech_endpoint_pytorch():
    response = client.post("/astromech", json={"crystalData": [[0.92, 0.12, 0.31, 0.09]], "model": "pytorch"})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "scores" in response.json()

def test_astromech_endpoint_invalid_model():
    response = client.post("/astromech", json={"crystalData": [[0.92, 0.12, 0.31, 0.09]], "model": "invalid_model"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid model type"}