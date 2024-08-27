import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
file_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/ENB2012_data.csv'
data = pd.read_csv(file_path)

# Assuming predictions are already made, we'll use Y1 and Y2 from the data for simplicity
# In a real scenario, you would load your model and make predictions here

# Normalize scores to a 0-100 scale
scaler = MinMaxScaler(feature_range=(0, 100))
data['Heating_Score'] = scaler.fit_transform(data[['Y1']])
data['Cooling_Score'] = scaler.fit_transform(data[['Y2']])

# Combine scores for an overall energy efficiency score (optional)
data['Energy_Efficiency_Score'] = (data['Heating_Score'] + data['Cooling_Score']) / 2

# Display rated data
# print(data[['Y1', 'Heating_Score', 'Y2', 'Cooling_Score', 'Energy_Efficiency_Score']].head())

# Save the rated data to a new CSV file
data.to_csv('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/EnergyEfficiencyPrediction/energy_efficiency_rated.csv', index=False)