import requests
import pytest
from fastapi.testclient import TestClient
import sys, os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app import app

client = TestClient(app)

class TestStationService:

    def test_successful_single_sklearn_response(self):
        crystal_data = [[0.92, 0.12, 0.31, 0.09]]
        input_data = {
            'crystalData': crystal_data
        }

        response = client.post("/sklearn", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_successful_batch_sklearn_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.7312, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = client.post("/sklearn", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_fail_single_sklearn_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09, 0.012],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = client.post("/sklearn", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_fail_batch_sklearn_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = client.post("/sklearn", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_successful_single_pytorch_response(self):
        crystal_data = [[0.92, 0.12, 0.31, 0.09]]
        input_data = {
            'crystalData': crystal_data
        }

        response = client.post("/pytorch", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_successful_batch_pytorch_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.7312, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data
        }
        response = client.post("/pytorch", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_fail_single_pytorch_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09, 0.012],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = client.post("/pytorch", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_fail_batch_pytorch_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = client.post("/pytorch", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_successful_single_astromech_response(self):
        crystal_data = [[0.92, 0.12, 0.31, 0.09]]
        input_data = {
            'crystalData': crystal_data,
            'model': 'sklearn'
        }

        response = client.post("/astromech", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_successful_batch_astromech_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.7312, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data,
            'model': 'sklearn'
        }

        response = client.post("/astromech", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_fail_single_astromech_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09, 0.012],
            ]
        input_data = {
            'crystalData': crystal_data,
            'model': 'sklearn'
        }

        response = client.post("/astromech", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_fail_batch_astromech_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data,
            'model': 'sklearn'
        }

        response = client.post("/astromech", json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_fail_model_astromech_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data,
            'model': 'notexistent'
        }

        response = client.post("/astromech", json=input_data)
        
        status_code = response.status_code
        body = response.json()
        assert status_code == 400