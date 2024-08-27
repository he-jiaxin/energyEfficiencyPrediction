import pytest
import os
import sys
import io
from flask import Flask, request
from unittest.mock import patch, MagicMock
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app, heat_load_model, cool_load_model

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_model_loading():
    assert heat_load_model is not None, "Heat load model should be loaded"
    assert cool_load_model is not None, "Cool load model should be loaded"

def test_missing_file_part(client):
    response = client.post('/segment', data={})
    assert response.status_code == 400
    assert 'No file part' in response.get_json()['error']

def test_empty_filename(client):
    data = {
        'input_image': (io.BytesIO(b""), '')  # Empty filename
    }
    response = client.post('/segment', content_type='multipart/form-data', data=data)
    assert response.status_code == 400
    assert 'No selected file' in response.get_json()['error']

@patch('app.os.makedirs')
@patch('app.os.path.exists', return_value=False)
def test_directory_creation(mock_exists, mock_makedirs, client):
    data = {
        'input_image': (io.BytesIO(b"fake_image_content"), 'testfile.png')
    }
    client.post('/segment', content_type='multipart/form-data', data=data)
    mock_makedirs.assert_called_once()


@patch('app.subprocess.run')
def test_command_construction(mock_subprocess, client):
    mock_subprocess.return_value = MagicMock(stdout='output.csv\n')
    data = {
        'input_image': (io.BytesIO(b"fake_image_content"), 'testfile.png')
    }
    response = client.post('/segment', content_type='multipart/form-data', data=data)
    assert mock_subprocess.call_count == 2  # Adjust if multiple calls are expected



def test_text_to_speech_input_validation(client):
    response = client.post('/api/text-to-speech', json={})
    assert response.status_code == 400
    assert 'Text is required' in response.get_json()['error']


def test_valid_file_upload(client):
    data = {
        'input_image': (io.BytesIO(b"fake_image_content"), 'testfile.png')
    }
    response = client.post('/segment', content_type='multipart/form-data', data=data)
    assert response.status_code == 500  # Expecting failure due to the subprocess
    assert 'Segmentation failed' in response.get_json()['error']

def test_invalid_file_upload(client):
    data = {
        'input_image': (io.BytesIO(b"Not an image"), 'testfile.txt')
    }
    response = client.post('/segment', content_type='multipart/form-data', data=data)
    assert response.status_code == 500
    assert 'Segmentation failed' in response.get_json().get('error', '')

def test_empty_text_for_tts(client):
    response = client.post('/api/text-to-speech', json={'text': ''})
    assert response.status_code == 400

def test_valid_tts_request(client):
    response = client.post('/api/text-to-speech', json={'text': 'Hello, world!'})
    assert response.status_code == 200
    assert response.mimetype == 'audio/wav'
