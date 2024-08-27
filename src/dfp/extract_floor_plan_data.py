import cv2
import pytesseract
import re
import numpy as np
from PIL import Image
import pandas as pd
from typing import Dict, Tuple, List
import os
import sys
from pdf2image import convert_from_path

# Dictionary for common room names and unique identifiers
room_name_dict = {
    'bedroom': 1,
    'ensuite': 2,
    'bathroom': 3,
    'kitchen': 4,
    'reception': 5,
    'living room': 6,
    'dining room': 7,
    'corridor': 8,
    'balcony': 9,
    'str': 10,    # Added Str
    'wrd': 11     # Added Wrd
}

# Dictionary for attribute names and unique identifiers
attribute_dict = {
    'ceiling height': 'ceiling_height',
    'glazing area': 'glazing_area',
    'orientation': 'orientation',
    'glazing distribution': 'glazing_distribution'
}

def load_image(image_path: str) -> Image.Image:
    if image_path.lower().endswith('.pdf'):
        return convert_pdf_to_image(image_path)
    else:
        return Image.open(image_path).convert('RGB')

def convert_pdf_to_image(pdf_path: str) -> Image.Image:
    # Convert PDF to a single image
    images = convert_from_path(pdf_path)
    if images:
        return images[0].convert('RGB')
    else:
        raise ValueError("PDF conversion failed or the PDF is empty.")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    binary = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)
    return binary

def segment_image(image: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 15:
            segment = image[y:y+h, x:x+w]
            segments.append(segment)
    return segments

def extract_text_from_segments(segments: List[np.ndarray]) -> str:
    text = ""
    for segment in segments:
        custom_config = r'--oem 3 --psm 6'
        segment_text = pytesseract.image_to_string(segment, config=custom_config)
        text += segment_text + "\n"
    return text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s*.x%]', '', text)  # Keep numbers, letters, spaces, periods, 'x', and '%'
    # text = re.sub(r'\s+', ' ', text)
    return text

def correct_dimensions(dimensions: Tuple[float, float]) -> Tuple[float, float]:
    # Check if dimensions seem to be misinterpreted by the factor of 10
    length, width = dimensions
    if length > 10:
        length /= 10
    if width > 10:
        width /= 10
    return length, width

def process_text(text: str) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    room_details = {}
    lines = text.split('\n')
    
    dimension_pattern = re.compile(r'(\d+(\.\d+)?)\s*x\s*(\d+(\.\d+)?)')
    room_name_pattern = re.compile('|'.join(room_name_dict.keys()), re.IGNORECASE)
    attribute_pattern = re.compile(r'(ceiling\s*height|glazing\s*area|orientation|glazing\s*distribution)', re.IGNORECASE)
    
    current_room_label = None
    room_labels = []
    dimensions = []
    room_count_tracker = {}

    # Default values
    attributes = {
        'ceiling_height': 2.8,
        'glazing_area': 0.50,
        'orientation': 2,
        'glazing_distribution': 1
    }

    pending_attributes = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        cleaned_line = clean_text(line)

        # Check if there are pending attributes
        if pending_attributes:
            values = cleaned_line.split()
            for idx, attribute in enumerate(pending_attributes):
                if idx < len(values):
                    attribute_value = float(values[idx])
                    # Fix for ceiling height
                    if attribute == 'ceiling_height' and attribute_value > 10:
                        attribute_value = attribute_value / 10
                    attributes[attribute] = attribute_value
                    print(f"Detected {attribute}: {attribute_value}")

            pending_attributes = []
            continue

        room_name_matches = list(room_name_pattern.finditer(cleaned_line))
        dimension_matches = list(dimension_pattern.finditer(cleaned_line))
        attribute_matches = list(attribute_pattern.finditer(cleaned_line))

        if room_name_matches or dimension_matches:
            for room_name_match in room_name_matches:
                room_name = room_name_match.group(0).strip().lower()
                if room_name in room_count_tracker:
                    room_count_tracker[room_name] += 1
                else:
                    room_count_tracker[room_name] = 1
                current_room_label = f"{room_name}_{room_count_tracker[room_name]}"
                room_labels.append(current_room_label)

            for dimension_match in dimension_matches:
                length = float(dimension_match.group(1))
                width = float(dimension_match.group(3))
                dimensions.append((length, width))

            if room_labels and dimensions:
                label = room_labels.pop(0)
                dimension = dimensions.pop(0)
                length, width = correct_dimensions(dimension)
                room_details[label] = (length, width)

        for attribute_match in attribute_matches:
            attribute_name = attribute_dict[attribute_match.group(1).strip().lower()]
            if attribute_name in attributes:
                pending_attributes.append(attribute_name)

    return room_details, attributes

def extract_floor_data(image_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # print(f"Starting extraction process for image: {image_path}")
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image)
    segments = segment_image(preprocessed_image)
    text = extract_text_from_segments(segments)
    
    # print("Extracted text data:")
    # print(text)
    
    room_details, attributes = process_text(text)
    
    # Convert to DataFrame for better accuracy checking
    df = pd.DataFrame(room_details.items(), columns=["Label", "Dimensions"])
    df[['Length', 'Width']] = pd.DataFrame(df['Dimensions'].tolist(), index=df.index)
    df.drop(columns=['Dimensions'], inplace=True)
    return df, attributes

def calculate_wall_area(df: pd.DataFrame, ceiling_height: float) -> pd.Series:
    wall_area = (df['Length'] * ceiling_height + df['Width'] * ceiling_height) * 2
    return wall_area

def calculate_number_of_rooms(df: pd.DataFrame) -> int:
    return len(df)

def calculate_total_floor_area(df: pd.DataFrame) -> float:
    total_floor_area = (df['Length'] * df['Width']).sum()
    return total_floor_area

def calculate_total_volume(df: pd.DataFrame, ceiling_height: float) -> float:
    total_volume = (df['Length'] * df['Width'] * ceiling_height).sum()
    return total_volume

def calculate_total_surface_area(df: pd.DataFrame, ceiling_height: float) -> float:
    total_wall_area = calculate_wall_area(df, ceiling_height).sum()
    total_ceiling_area = calculate_total_floor_area(df)
    total_surface_area = total_wall_area + total_ceiling_area
    return total_surface_area

def calculate_relative_compactness(total_volume: float, total_surface_area: float) -> float:
    if total_surface_area != 0:
        relative_compactness = (total_volume / total_surface_area) + 0.3
    else:
        relative_compactness = 0
    return relative_compactness

def save_calculations_to_csv(df: pd.DataFrame, attributes: Dict[str, float], csv_path: str):
    ceiling_height = attributes['ceiling_height']
    glazing_area = attributes['glazing_area']
    orientation = attributes['orientation']
    glazing_distribution = attributes['glazing_distribution']
    
    total_volume = calculate_total_volume(df, ceiling_height)
    total_surface_area = calculate_total_surface_area(df, ceiling_height)
    relative_compactness = calculate_relative_compactness(total_volume, total_surface_area)
    
    calculations = {
        'Relative Compactness': round(relative_compactness, 2),  # Relative Compactness
        'Wall Area': round(calculate_wall_area(df, ceiling_height).sum(), 2),  # Wall Area
        'Roof Area': round(calculate_total_floor_area(df)*1.1, 2), # 'Roof Area' Justify for overhangs, eaves
        'Overall Height': round(ceiling_height, 2), # Overall Height
        # 'Number of Rooms': calculate_number_of_rooms(df),  
        'Orientation': round(orientation, 2),  # Orientation
        'Glazing Area': round(glazing_area, 2),  # Glazing Area
        'Glazing Distribution': round(glazing_distribution, 2)  # Glazing Distribution
    }

    calculations_df = pd.DataFrame([calculations])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    calculations_df.to_csv(csv_path, index=False)

# Add this block at the end to make the script runnable
if __name__ == "__main__":
    if len(sys.argv) < 4:
        # print("Usage: script.py <image_path> <data_csv_path> <calculations_csv_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    data_csv_path = sys.argv[2]
    calculations_csv_path = sys.argv[3]

    # Extract floor data from the image
    df, attributes = extract_floor_data(image_path)
    # debug print

    # print(df)
    # print(attributes)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(data_csv_path), exist_ok=True)

    # Save the extracted data to a CSV
    df.to_csv(data_csv_path, index=False)

    # Save the calculations to a different CSV
    save_calculations_to_csv(df, attributes, calculations_csv_path)
    # print(f"Calculations saved to {calculations_csv_path}")