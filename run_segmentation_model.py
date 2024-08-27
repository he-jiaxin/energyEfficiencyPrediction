import argparse
import csv
import gc
import os
import sys
import subprocess
from typing import List, Tuple, Dict
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from pdf2image import convert_from_path

from src.dfp.deploy import (
    load_image, init, predict, post_process, colorize, convert_one_hot_to_image, overwrite_args_with_toml
)
from src.dfp.data import convert_one_hot_to_image
from src.dfp.net import deepfloorplanModel
from src.dfp.net_func import deepfloorplanFunc
from src.dfp.utils.rgb_ind_convertor import (
    floorplan_boundary_map,
    floorplan_fuse_map,
    ind2rgb,
)
from src.dfp.utils.util import fill_break_line, flood_fill, refine_room_region

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def load_extracted_data(csv_path: str) -> Tuple[Dict[str, Tuple[int, int, int, int, float, float]], Dict[str, List[Tuple[int, int, int, int]]]]:
    room_labels = {}
    window_positions = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 3:  # Room label row
                try:
                    room_labels[row[0]] = (0, 0, 0, 0, float(row[1]), float(row[2]))
                except ValueError:
                    continue
            elif len(row) == 5:  # Window position row
                try:
                    if row[0] not in window_positions:
                        window_positions[row[0]] = []
                    window_positions[row[0]].append((int(row[1]), int(row[2]), int(row[3]), int(row[4])))
                except ValueError:
                    continue
    return room_labels, window_positions

def main(config: argparse.Namespace):
    output_dir = "/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate consistent filenames based on input image
    input_filename = os.path.splitext(os.path.basename(config.image))[0]
    
    # Set output paths if not provided
    if config.csv_output is None:
        config.csv_output = os.path.join(output_dir, f'{input_filename}_data.csv')
    if config.calculations_output is None:
        config.calculations_output = os.path.join(output_dir, f'{input_filename}_calculation.csv')

    # Ensure that the csv files are generated
    run_extract_floor_plan(config.image, config.csv_output, config.calculations_output)

    model, img, shp = init(config)
    if config.loadmethod == "tflite":
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]["index"], img)
        model.invoke()
        ri, cwi = 0, 1
        if config.tfmodel == "func":
            ri, cwi = 1, 0
        logits_r = model.get_tensor(output_details[ri]["index"])
        logits_cw = model.get_tensor(output_details[cwi]["index"])
        logits_cw = tf.convert_to_tensor(logits_cw)
        logits_r = tf.convert_to_tensor(logits_r)
    else:
        if config.tfmodel == "func":
            logits_r, logits_cw = model.predict(img)
        elif config.tfmodel == "subclass":
            if config.loadmethod == "log":
                logits_cw, logits_r = predict(model, img, shp)
            elif config.loadmethod == "pb" or config.loadmethod == "none":
                logits_r, logits_cw = model(img)
    
    logits_r = tf.image.resize(logits_r, shp[:2])
    logits_cw = tf.image.resize(logits_cw, shp[:2])
    r = convert_one_hot_to_image(logits_r)[0].numpy()
    cw = convert_one_hot_to_image(logits_cw)[0].numpy()

    # Load the extracted data
    room_labels, window_positions = load_extracted_data(config.csv_output)

    if not config.colorize and not config.postprocess:
        cw[cw == 1] = 9
        cw[cw == 2] = 10
        r[cw != 0] = 0
        result = (r + cw).squeeze()
    elif config.colorize and not config.postprocess:
        r_color, cw_color = colorize(r.squeeze(), cw.squeeze())
        result = r_color + cw_color
    else:
        newr, newcw = post_process(r, cw, shp)
        if not config.colorize and config.postprocess:
            newcw[newcw == 1] = 9
            newcw[newcw == 2] = 10
            newr[newcw != 0] = 0
            result = newr.squeeze() + newcw
        else:
            newr_color, newcw_color = colorize(newr.squeeze(), newcw.squeeze())
            result = newr_color + newcw_color

    image_output_path = os.path.join(output_dir, f"{input_filename}_result.jpg")

    if config.save:
        mpimg.imsave(image_output_path, result.astype(np.uint8))

    return result
def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tfmodel", type=str, default="subclass", choices=["subclass", "func"])
    p.add_argument("--image", type=str, default="resources/30939153.jpg")
    p.add_argument("--weight", type=str, default="log/store/G")
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--colorize", action="store_true")
    p.add_argument("--loadmethod", type=str, default="log", choices=["log", "tflite", "pb", "none"])
    p.add_argument("--save", type=str)
    p.add_argument("--csv_output", type=str, help="Path to save the output CSV file")
    p.add_argument("--calculations_output", type=str, help="Path to save the calculations CSV file")
    p.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor to convert from pixels to another unit")
    p.add_argument("--unit", type=str, default="pixels", help="Unit of measurement for the coordinates and dimensions")
    p.add_argument("--feature-channels", type=int, action="store", default=[256, 128, 64, 32], nargs=4)
    p.add_argument("--backbone", type=str, default="vgg16", choices=["vgg16", "resnet50", "mobilenetv1", "mobilenetv2"])
    p.add_argument("--feature-names", type=str, action="store", nargs=5, default=["block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"])
    p.add_argument("--tomlfile", type=str, default=None)
    return p.parse_args(args)

def run_extract_floor_plan(image_path, data_csv_path, calculations_csv_path):
    script_path = "src/dfp/extract_floor_plan_data.py"  
    command = [
        sys.executable, script_path, image_path, data_csv_path, calculations_csv_path
    ]

    print(f"Running {script_path} with image_path: {image_path}, data_csv_path: {data_csv_path}, calculations_csv_path: {calculations_csv_path}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Subprocess output: {result.stdout}")
        if result.stderr:
            print(f"Subprocess error: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        sys.exit(1)

def deploy_plot_res(result: np.ndarray):
    print(result.shape)
    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = overwrite_args_with_toml(args)
    
    # Define the output directory and ensure it exists
    output_dir = "/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the paths for the CSV outputs using the input filename
    input_filename = os.path.splitext(os.path.basename(args.image))[0]
    args.csv_output = os.path.join(output_dir, f"{input_filename}_data.csv")
    args.calculations_output = os.path.join(output_dir, f"{input_filename}_calculation.csv")
    
    # Run the extract_floor_plan_data script
    run_extract_floor_plan(args.image, args.csv_output, args.calculations_output)
    
    # Proceed with your main function logic
    result = main(args)
    deploy_plot_res(result)
