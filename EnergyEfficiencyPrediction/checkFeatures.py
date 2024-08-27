import joblib
import numpy as np
import pandas as pd

# Paths to your models
heating_model_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_heating_corrected.joblib'
cooling_model_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_cooling_corrected.joblib'

# Feature names (adjust if necessary)
feature_names = ['relative_compactness', 'wall_area', 'roof_area', 'overall_height', 
                 'orientation', 'glazing_area', 'glazing_area_distribution']

def check_feature_importance(model_path, model_name):
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Check for feature_importances_
        if hasattr(model, 'feature_importances_'):
            print(f"{model_name} model has feature_importances_")
            
            # Create a DataFrame of feature importances
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            })
            
            # Sort importances in descending order
            importances = importances.sort_values('Importance', ascending=False).reset_index(drop=True)
            
            # Print the importances
            print(f"{model_name} feature importances:")
            print(importances)
            print("\n")
        else:
            print(f"{model_name} model does not have feature_importances_")
    except Exception as e:
        print(f"Error loading or checking {model_name} model: {str(e)}")

# Check both models
check_feature_importance(heating_model_path, "Heating")
check_feature_importance(cooling_model_path, "Cooling")