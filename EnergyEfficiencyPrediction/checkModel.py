from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the best models
heating_model_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_heating_corrected.joblib'
model_heating = load(heating_model_path)

cooling_model_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/model_cooling_corrected.joblib'
model_cooling = load(cooling_model_path)

# Load your dataset
data = pd.read_csv('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/ENB2012_data.csv')
data.columns = ['relative_compactness', 'wall_area', 'roof_area', 'overall_height', 'orientation',
                'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']


# Select features and target variable
# Ensure 'orientation' is included if it was used during training
X = data[['relative_compactness', 'wall_area', 'roof_area', 'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution']]
y1 = data['heating_load']
y2 = data['cooling_load']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)
X_train_cooling, X_test_cooling, y_train_cooling, y_test_cooling = train_test_split(X, y2, test_size=0.2, random_state=42)

# Make predictions using the heating model
y_pred_heating = model_heating.predict(X_test)

# Make predictions using the cooling model
y_pred_cooling = model_cooling.predict(X_test_cooling)

# Evaluate heating model performance
mae_heating = mean_absolute_error(y_test, y_pred_heating)
mse_heating = mean_squared_error(y_test, y_pred_heating)
r2_heating = r2_score(y_test, y_pred_heating)

print(f'Heating Model Performance:')
print(f'Mean Absolute Error (MAE): {mae_heating}')
print(f'Mean Squared Error (MSE): {mse_heating}')
print(f'R-squared (R2): {r2_heating}')

# Evaluate cooling model performance
mae_cooling = mean_absolute_error(y_test_cooling, y_pred_cooling)
mse_cooling = mean_squared_error(y_test_cooling, y_pred_cooling)
r2_cooling = r2_score(y_test_cooling, y_pred_cooling)

print(f'Cooling Model Performance:')
print(f'Mean Absolute Error (MAE): {mae_cooling}')
print(f'Mean Squared Error (MSE): {mse_cooling}')
print(f'R-squared (R2): {r2_cooling}')

# Print the input value and the output prediction for the first row
first_row = X.iloc[0].values.reshape(1, -1)
heating_prediction = model_heating.predict(first_row)
cooling_prediction = model_cooling.predict(first_row)

print("\nFirst row input values:")
print(X.iloc[0])

print("\nPredicted Heating Load for the first row:")
print(heating_prediction[0])

print("\nPredicted Cooling Load for the first row:")
print(cooling_prediction[0])

# Calculate predictions for the entire dataset and compare them with actual values
all_y_pred_heating = model_heating.predict(X)
all_y_pred_cooling = model_cooling.predict(X)

# Add predictions to the dataframe
data['predicted_heating_load'] = all_y_pred_heating
data['predicted_cooling_load'] = all_y_pred_cooling

print("\nComparison of actual and predicted heating loads for the entire dataset:")
print(data[['heating_load', 'predicted_heating_load']].head())

print("\nComparison of actual and predicted cooling loads for the entire dataset:")
print(data[['cooling_load', 'predicted_cooling_load']].head())