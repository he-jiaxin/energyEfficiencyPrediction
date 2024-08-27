import tensorflow as tf

def inspect_saved_model(modeldir):
    model = tf.keras.models.load_model(modeldir)
    model.summary()

# Call this function with your model directory (note: use the directory, not the .pb file)
inspect_saved_model('/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/model/store')
