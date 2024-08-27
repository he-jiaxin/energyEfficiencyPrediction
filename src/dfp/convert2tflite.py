import argparse
import os
import sys
import tempfile
from typing import List

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tqdm import tqdm

from .data import decodeAllRaw, loadDataset, preprocess
from .net import deepfloorplanModel
from .net_func import deepfloorplanFunc
from .train import train_step
from .utils.settings import overwrite_args_with_toml
from .utils.util import (
    print_model_weight_clusters,
    print_model_weights_sparsity,
)

def model_init(config: argparse.Namespace) -> tf.keras.Model:
    if config.loadmethod == "log":
        if config.tfmodel == "subclass":
            base_model = deepfloorplanModel(config=config)
            base_model.build((1, 512, 512, 3))
            raise ValueError("subclass and log are not convertible to tflite.")
        elif config.tfmodel == "func":
            base_model = deepfloorplanFunc(config=config)
        base_model.load_weights(config.modeldir)
    elif config.loadmethod == "pb":
        base_model = tf.keras.models.load_model(config.modeldir)
        # Ensure that the model layers are not trainable
        for layer in base_model.layers:
            layer.trainable = False
    return base_model

def converter(config: argparse.Namespace):
    model = model_init(config)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if config.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(config.tflitedir, "wb") as f:
        f.write(tflite_model)

def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Convert a TensorFlow model to TFLite format.')
    p.add_argument('--modeldir', type=str, required=True, help='Directory where the model is stored.')
    p.add_argument('--tflitedir', type=str, required=True, help='Directory to save the TFLite model.')
    p.add_argument('--loadmethod', type=str, choices=['log', 'pb'], required=True, help='Method to load the model.')
    p.add_argument('--quantize', action='store_true', help='Flag to apply quantization.')
    p.add_argument('--tfmodel', type=str, choices=['subclass', 'func'], required=True, help='Type of TensorFlow model.')
    p.add_argument('--feature-channels', type=int, nargs='+', required=True, help='Feature channels for the model.')
    p.add_argument('--backbone', type=str, choices=['vgg16', 'mobilenetv1', 'mobilenetv2', 'resnet50'], required=True, help='Backbone model architecture.')
    p.add_argument('--feature-names', type=str, nargs='+', required=True, help='Names of the feature layers.')
    return p.parse_args(args)

def prune(config: argparse.Namespace):
    base_model = model_init(config)
    print(base_model.summary())
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)
    print(model_for_pruning.summary())
    dataset = loadDataset()
    optimizer = tf.keras.optimizers.Adam()
    log_dir = tempfile.mkdtemp()
    unused_arg = -1
    epochs = 4
    batches = 8

    model_for_pruning.optimizer = optimizer
    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model_for_pruning)
    log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)  # Log sparsity and other metrics in Tensorboard.
    log_callback.set_model(model_for_pruning)

    step_callback.on_train_begin()  # run pruning callback
    for _ in range(epochs):
        log_callback.on_epoch_begin(epoch=unused_arg)  # run pruning callback
        for data in tqdm(list(dataset.batch(batches))):
            step_callback.on_train_batch_begin(batch=unused_arg)  # run pruning callback
            img, bound, room = decodeAllRaw(data)
            img, bound, room, hb, hr = preprocess(img, bound, room)
            _, _, loss_value, _, _ = train_step(model_for_pruning, optimizer, img, hr, hb)
        step_callback.on_epoch_end(batch=unused_arg)  # run pruning callback

    print(f"log directory: {log_dir}...")
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    print_model_weights_sparsity(model_for_export)
    model_for_export.save(log_dir + "/prune")

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    pruned_tflite_model = converter.convert()

    os.system(f"mkdir -p {log_dir}/tflite")
    with open(log_dir + "/tflite/model.tflite", "wb") as f:
        f.write(pruned_tflite_model)

def cluster(config: argparse.Namespace):
    base_model = model_init(config)
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
        "number_of_clusters": 8,
        "cluster_centroids_init": CentroidInitialization.DENSITY_BASED,
    }

    def apply_clustering_to_conv2d(layer):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Conv2DTranspose):
            return cluster_weights(layer, **clustering_params)
        return layer

    clustered_model = tf.keras.models.clone_model(
        base_model,
        clone_function=apply_clustering_to_conv2d,
    )

    print(clustered_model.summary())

    dataset = loadDataset()
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(4):
        print("[INFO] Epoch {}".format(epoch))
        for data in tqdm(list(dataset.batch(8))):
            img, bound, room = decodeAllRaw(data)
            img, bound, room, hb, hr = preprocess(img, bound, room)
            _, _, loss_value, _, _ = train_step(clustered_model, optimizer, img, hr, hb)

    log_dir = tempfile.mkdtemp()
    print("log directory: " + log_dir)
    final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
    print_model_weight_clusters(final_model)
    final_model.save(log_dir + "/cluster")

    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    clustered_tflite_model = converter.convert()

    os.system(f"mkdir -p {log_dir}/tflite")
    with open(log_dir + "/tflite/model.tflite", "wb") as f:
        f.write(clustered_tflite_model)

def quantization_aware_training(config: argparse.Namespace):
    base_model = model_init(config)

    def apply_quantization_to_conv2D(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated_model = tf.keras.models.clone_model(
        base_model,
        clone_function=apply_quantization_to_conv2D,
    )
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    print(quant_aware_model.summary())

    dataset = loadDataset()
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(4):
        print("[INFO] Epoch {}".format(epoch))
        for data in tqdm(list(dataset.batch(8))):
            img, bound, room = decodeAllRaw(data)
            img, bound, room, hb, hr = preprocess(img, bound, room)
            _, _, loss_value, _, _ = train_step(quant_aware_model, optimizer, img, hr, hb)

    log_dir = tempfile.mkdtemp()
    print("log directory: " + log_dir)

    # Save the quantization-aware trained model
    quant_aware_model.save(os.path.join(log_dir, 'quant_aware_model'))

    # Convert the quantization-aware trained model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(log_dir, 'quant_aware_model'))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()

    os.makedirs(os.path.join(log_dir, "tflite"), exist_ok=True)
    with open(os.path.join(log_dir, "tflite", "model.tflite"), "wb") as f:
        f.write(quantized_tflite_model)

    print(f"TFLite model saved to: {os.path.join(log_dir, 'tflite', 'model.tflite')}")

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    converter(args)