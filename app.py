from flask import Flask, Response, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import logging
import os
import sys
import subprocess
import csv
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import subprocess

# IBM
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
process = {}


# logging.debug("Loading heat load model...")
heat_load_model = joblib.load('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_heating_corrected.joblib')
cool_load_model = joblib.load('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_cooling_corrected.joblib')

DATASET_PATH = 'EnergyEfficiencyPrediction/dataset.csv'
OUTPUT_DIR = 'output'



# IBM Watson Text to Speech configuration
authenticator = IAMAuthenticator('vRwaf_O5fKcNFd_UG6lHvxVuGBlSbnWTiRqIUKlAI-4j')
text_to_speech = TextToSpeechV1(authenticator=authenticator)
text_to_speech.set_service_url('https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/e3894273-38c5-4b97-937c-f84edbbf3661')

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech_api():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        response = text_to_speech.synthesize(
            text,
            voice='en-US_AllisonV3Voice',
            accept='audio/wav'
        ).get_result()

        audio_content = response.content

        # Send back the audio file as a response
        return Response(audio_content, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/segment', methods=['POST'])
def segment():
    logging.debug("Received a request to /segment")
    
    if 'input_image' not in request.files:
        logging.error('No file part found in the request')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['input_image']
    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    
    # Ensure the 'uploads' directory exists
    upload_directory = os.path.join(os.path.dirname(__file__), 'uploads')
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)
    
    # Save the file to the 'uploads' directory
    input_image_path = os.path.join(upload_directory, file.filename)
    logging.debug(f"Saving file to: {input_image_path}")
    file.save(input_image_path)

    logging.debug(f"Segmenting image: {input_image_path}")

    # Derive the output CSV path from the input image name
    input_image_name = os.path.splitext(os.path.basename(input_image_path))[0]
    csv_output_path = os.path.join('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output', f'{input_image_name}_calculation.csv')
    logging.debug(f"CSV output path: {csv_output_path}")

    # Call run_segmentation_model.py as a subprocess
    cmd = [
        sys.executable,
        'run_segmentation_model.py',
        '--image', input_image_path,
        '--weight', '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/log/store/G',
        '--postprocess',
        '--colorize',
        '--save', 'output.jpg',
        '--loadmethod', 'log'
    ]

    try:
        # logging.debug(f"Running segmentation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # logging.debug(f"Segmentation output: {result.stdout}")
        
        # Assume the last line of output is the path to the result image
        result_path = result.stdout.strip().split('\n')[-1]
        # logging.debug(f"Segmentation result path: {result_path}")
        
        # **Modification**: Call the predict function using the generated CSV path
        prediction = predict_from_csv(csv_output_path)
        # logging.debug(f"Prediction result: {prediction}")
        
        # Generate heatmap based on the prediction
        heatmap_result = generate_heatmap(input_image_path)
        # logging.debug(f"Generated heatmap: {heatmap_result}")
        
        logging.debug("Segmentation, prediction, and heatmap generation completed.")
        return jsonify({
            'segmentation_result': result_path, 
            'prediction': {
                'heat_load_prediction': prediction['heat_load_prediction'],
                'cool_load_prediction': prediction['cool_load_prediction'],
                'heat_load_feature_importance': prediction['heat_load_feature_importance'],
                'cool_load_feature_importance': prediction['cool_load_feature_importance']
            },
            'heatmap': heatmap_result
        })
    except subprocess.CalledProcessError as e:
        logging.error(f"Segmentation failed: {e.stderr}")
        return jsonify({'error': 'Segmentation failed'}), 500
    except Exception as e:
        logging.error(f"Failed to process prediction: {e}")
        return jsonify({'error': 'Prediction processing failed'}), 500




def get_feature_importance(model, feature_names):
    try:
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
        # logging.debug(f"Feature importance calculated: {feature_importance.to_dict(orient='records')}")
        return feature_importance.to_dict(orient='records')
    except Exception as e:
        # logging.error(f"Error in get_feature_importance: {str(e)}")
        return []



def predict_from_csv(csv_path):
    try:
        features = read_csv_data(csv_path)
        if not features:
            return {'error': 'No data found in CSV file'}

        features = np.array(features[0]).reshape(1, -1)
        
        feature_names = ['relative_compactness', 'wall_area', 'roof_area', 'overall_height',
                         'orientation', 'glazing_area', 'glazing_area_distribution']

        heat_prediction = heat_load_model.predict(features)
        cool_prediction = cool_load_model.predict(features)

        heat_importance = get_feature_importance(heat_load_model, feature_names)
        cool_importance = get_feature_importance(cool_load_model, feature_names)

        # logging.debug(f"Heat Load Feature Importance: {heat_importance}")
        # logging.debug(f"Cool Load Feature Importance: {cool_importance}")

        return {
            'heat_load_prediction': heat_prediction.tolist(),
            'cool_load_prediction': cool_prediction.tolist(),
            'heat_load_feature_importance': heat_importance,
            'cool_load_feature_importance': cool_importance
        }
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return {'error': f'An error occurred during prediction: {str(e)}'}
    

@app.route('/get-image/<filename>', methods=['GET'])
def get_image(filename):
    try:
        # Log the incoming request
        app.logger.info(f"Received request for filename: {filename}")
        
        if filename.endswith('_heatmap.png') or filename.endswith('_coolmap.png'):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(file_path):
                return send_from_directory(OUTPUT_DIR, filename)
            else:
                app.logger.error(f"File not found: {filename}")
                return jsonify({'error': 'File not found'}), 404
        else:
            app.logger.error(f"Invalid filename format: {filename}")
            return jsonify({'error': 'Invalid filename format'}), 400
    except Exception as e:
        app.logger.exception("Error occurred while fetching image")
        return jsonify({'error': str(e)}), 500


# Function to read CSV data
def read_csv_data(csv_path):
    features = []
    try:
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                features.append([float(val) for val in row])
    except Exception as e:
        logging.error(f"Failed to read CSV data: {e}")
    return features

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    csv_path = data.get('csv_path')
    
    if not csv_path:
        return jsonify({'error': 'CSV path is required'}), 400

    try:
        prediction = predict_from_csv(csv_path)
        # logging.debug(f"Prediction result: {prediction}")
        return jsonify(prediction)
    except Exception as e:
        logging.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def generate_heatmap(layout_img_path):
    try:
        result = subprocess.run([
            sys.executable, 'heatmap.py',
            '--layout_img_path', layout_img_path
        ], check=True, capture_output=True, text=True)
        # logging.debug(f"Heatmap generation output: {result.stdout}")
        
        heatmap_path = result.stdout.strip().split('\n')[-1]
        return heatmap_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Heatmap generation failed: {e.stderr}")
        return None

@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.json
    
    csv_path = data.get('csv_path')
    heat_load = data.get('heat_load')
    cool_load = data.get('cool_load')

    if not csv_path:
        return jsonify({'error': 'CSV path is required'}), 400
    if heat_load is None or cool_load is None:
        return jsonify({'error': 'Both heat_load and cool_load are required'}), 400

    # Extract features from the CSV file using the provided function
    features = read_csv_data(csv_path)
    if not features or len(features[0]) != 7:
        return jsonify({'error': 'Failed to extract valid features from the CSV file'}), 400

    new_features = np.array(features[0]).reshape(1, -1)  # Use the first row of features

    try:
        heat_load = float(heat_load)
        cool_load = float(cool_load)
    except ValueError:
        return jsonify({'error': 'heat_load and cool_load must be numeric values'}), 400

    # Append new data to CSV
    new_row = np.hstack((new_features[0], [heat_load, cool_load]))
    try:
        with open(DATASET_PATH, 'a') as f:
            np.savetxt(f, [new_row], delimiter=',')
    except Exception as e:
        return jsonify({'error': f'Failed to append data to CSV: {str(e)}'}), 500

    # Update models with new data
    try:
        if hasattr(heat_load_model, 'partial_fit') and hasattr(cool_load_model, 'partial_fit'):
            heat_load_model.partial_fit(new_features, [heat_load])
            cool_load_model.partial_fit(new_features, [cool_load])
        else:
            # If partial_fit is not available, retrain on the entire dataset
            df = pd.read_csv(DATASET_PATH)
            features = df.iloc[:, :-2].values
            heat_targets = df.iloc[:, -2].values
            cool_targets = df.iloc[:, -1].values
            heat_load_model.fit(features, heat_targets)
            cool_load_model.fit(features, cool_targets)
    except Exception as e:
        return jsonify({'status': 'retraining failed', 'error': str(e)}), 500

    # Evaluate new models using standard metrics
    try:
        mae_heat = mean_absolute_error(heat_targets, heat_load_model.predict(features))
        mse_heat = mean_squared_error(heat_targets, heat_load_model.predict(features))
        r2_heat = r2_score(heat_targets, heat_load_model.predict(features))

        mae_cool = mean_absolute_error(cool_targets, cool_load_model.predict(features))
        mse_cool = mean_squared_error(cool_targets, cool_load_model.predict(features))
        r2_cool = r2_score(cool_targets, cool_load_model.predict(features))

        response = {
            'status': 'models retrained successfully',
            'fitted_data': {
                'new_features': new_features.tolist(),
                'heat_load': heat_load,
                'cool_load': cool_load,
                'mae_heat': mae_heat,
                'mse_heat': mse_heat,
                'r2_heat': r2_heat,
                'mae_cool': mae_cool,
                'mse_cool': mse_cool,
                'r2_cool': r2_cool
            }
        }
    except Exception as e:
        return jsonify({'status': 'evaluation failed', 'error': str(e)}), 500

    # Check if models have stabilized before saving
    if r2_heat > 0.7 and r2_cool > 0.7:
        try:
            joblib.dump(heat_load_model, 'EnergyEfficiencyPrediction/model_heating_corrected.joblib')
            joblib.dump(cool_load_model, 'EnergyEfficiencyPrediction/model_cooling_corrected.joblib')
        except Exception as e:
            return jsonify({'status': 'model saving failed', 'error': str(e)}), 500

    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'endpoints': [
            {'path': '/segment', 'method': 'POST', 'description': 'Segment a floor plan image'},
            {'path': '/predict', 'method': 'POST', 'description': 'Make a heat and cool load prediction'},
            {'path': '/retrain', 'method': 'POST', 'description': 'Retrain the heat and cool load models'}
        ]
    })

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Flask app for segmentation and load prediction')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the Flask app on')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the Flask app on')
    parser.add_argument('--heat_load_model_path', type=str, default='/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_heating_corrected.joblib', help='Path to the heat load model')
    parser.add_argument('--cool_load_model_path', type=str, default='/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_cooling_corrected.joblib', help='Path to the cooling load model')
    
    args = parser.parse_args()
    
    heat_load_model = joblib.load(args.heat_load_model_path)
    cool_load_model = joblib.load(args.cool_load_model_path)
    
    app.run(host=args.host, port=args.port)