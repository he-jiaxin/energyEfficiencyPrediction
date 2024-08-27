import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import csv

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_text_to_speech_valid_text(client):
    response = client.post('/api/text-to-speech', json={'text': 'Hello, world!'})
    assert response.status_code == 200
    assert response.mimetype == 'audio/wav'


@patch('subprocess.run')
def test_segment_valid_image(mock_subprocess, client):
    # Simulate successful subprocess execution with realistic output
    mock_subprocess_instance = MagicMock()
    mock_subprocess_instance.returncode = 0
    mock_subprocess_instance.stdout = b'Successful segmentation\n'  # Correctly using bytes
    mock_subprocess.return_value = mock_subprocess_instance
    
    # Create a dummy CSV file with 7 features
    dummy_csv_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output/floor_plan_calculation.csv'
    os.makedirs(os.path.dirname(dummy_csv_path), exist_ok=True)
    with open(dummy_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation', 'glazing_area'])  # Expected headers
        writer.writerow([0.75, 650.0, 300.0, 200.0, 7.0, 2, 0.1])  # Dummy data
    
    if not os.path.exists('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/tests/floor_plan.png'):
        pytest.fail("Test data file 'floor_plan.png' is missing.")
    
    with open('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/tests/floor_plan.png', 'rb') as img:
        data = {
            'input_image': (img, 'floor_plan.png')
        }
        response = client.post('/segment', content_type='multipart/form-data', data=data)
    
    assert response.status_code == 500

def test_predict_valid_data_from_csv(client):
    # Define the CSV path that the application will read from
    csv_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output/floor_plan_calculation.csv'
    
    # Ensure the CSV file exists and contains valid data before running this test
    # You might want to create a fixture or setup method that prepares this file

    # Send the request with the CSV path
    response = client.post('/predict', json={"csv_path": csv_path})
    
    assert response.status_code == 200
    
    data = response.get_json()
    
    # Check predictions are in the response
    assert 'heat_load_prediction' in data
    assert 'cool_load_prediction' in data
    
    # Check feature importance is in the response
    assert 'heat_load_feature_importance' in data
    assert 'cool_load_feature_importance' in data
    
    # Verify the structure of the feature importance
    for importance in data['heat_load_feature_importance']:
        assert 'Feature' in importance
        assert 'Importance' in importance
    
    for importance in data['cool_load_feature_importance']:
        assert 'Feature' in importance
        assert 'Importance' in importance


from unittest.mock import patch

@patch("builtins.open", create=True)
def test_retrain_with_new_data(mock_open, client):
    mock_open.side_effect = FileNotFoundError("File not found")
    response = client.post('/retrain', json={
        'csv_path': '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/tests/new_training_data.csv',
        'heat_load': 20.5,
        'cool_load': 18.3
    })
    print(response.data)
    assert response.status_code == 400


def test_end_to_end_workflow(client):
    # Step 1: Segment an image
    # Mock the subprocess call used in the segmentation process
    with patch('subprocess.run') as mock_subprocess:
        mock_subprocess_instance = MagicMock()
        mock_subprocess_instance.returncode = 0
        
        # Simulate output as a string, mimicking a successful execution with a file path
        # Adjust the output format based on what the application expects (string or bytes)
        mock_subprocess_instance.stdout = "output/floor_plan_calculation.csv\n"  # As a string
        mock_subprocess.return_value = mock_subprocess_instance
        
        # Create a dummy CSV file with 7 features for the segmentation output
        dummy_csv_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output/floor_plan_calculation.csv'
        os.makedirs(os.path.dirname(dummy_csv_path), exist_ok=True)
        with open(dummy_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation', 'glazing_area'])
            writer.writerow([0.75, 650.0, 300.0, 200.0, 7.0, 2, 0.1])  # Dummy data
        
        # Ensure the image exists
        img_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/tests/floor_plan.png'
        if not os.path.exists(img_path):
            pytest.fail(f"Test image '{img_path}' is missing.")
    
        with open(img_path, 'rb') as img:
            data = {
                'input_image': (img, 'floor_plan.png')
            }
            response = client.post('/segment', content_type='multipart/form-data', data=data)
        
        assert response.status_code == 200  # Adjusted to expect a successful status code

    # Step 2: Predict using the segmented data
    response = client.post('/predict', json={"csv_path": dummy_csv_path})
    assert response.status_code == 200
    
    data = response.get_json()
    assert 'heat_load_prediction' in data
    assert 'cool_load_prediction' in data

    # Step 3: Retrain the model with new data
    # Assuming new_training_data.csv exists in the expected location
    training_csv_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/tests/new_training_data.csv'
    if not os.path.exists(training_csv_path):
        pytest.fail(f"Test training data file '{training_csv_path}' is missing.")
    
    response = client.post('/retrain', json={
        'csv_path': training_csv_path,
        'heat_load': 20.5,
        'cool_load': 18.3
    })
    assert response.status_code == 500