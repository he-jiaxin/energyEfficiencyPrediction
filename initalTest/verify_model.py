import tensorflow as tf

# Path to the saved model directory
model_path = '/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/model/store'

# Load the model
model = tf.keras.models.load_model(model_path)

# Display the model summary
model.summary()