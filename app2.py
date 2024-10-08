import streamlit as st
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output
import numpy as np
import pandas as pd
import cv2

# Configure Tesseract path if needed (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def draw_bboxes(image, results):
    img_draw = image.convert('RGB')
    draw = ImageDraw.Draw(img_draw)
    
    for result in results:
        bbox = result['box']
        text = result['text']
        draw.rectangle([bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']], outline='red', width=2)
        draw.text((bbox['x'], bbox['y'] - 10), text, fill=(255, 0, 0))  # Drawing the text above the box
    
    return img_draw

def crop_and_preprocess(image, y1=345, y2=795):
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Crop the image
    cropped = gray[y1:y2, :]
    
    return Image.fromarray(cropped)

def run_tesseract(image):
    # Convert image to OpenCV format
    img_np = np.array(image)
    
    # Run Tesseract OCR
    data = pytesseract.image_to_data(img_np, lang='ara', output_type=Output.DICT)
    
    results = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:  # Only consider non-empty text
            bbox = {
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i]
            }
            results.append({'text': text, 'box': bbox})
    
    return results

# Set up Streamlit
st.title("Arabic Text Extraction with Tesseract OCR")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    cropped_image = crop_and_preprocess(image)
    st.image(cropped_image, caption='Cropped Image', use_column_width=True)
    
    # Extract text button
    if st.button("Extract Text with Tesseract"):
        with st.spinner("Extracting text..."):
            results = run_tesseract(cropped_image)
            
            # Extract fields based on indicators
            extracted_data = []
            indicator_texts = {
                "Name of requester": "نعم انا",
                "Nationality of requester": "الجنسية",
                "National ID of requester": "رقم الإثبات",
                "Phone number of requester": "رقم اتصال",
                "Name of deceased": "اقر باني استلمت جثمان المتوفى",
                "Nationality of deceased": "الجنسية",
                "National ID of deceased": "رقم الإثبات"
            }

            # Extract text near indicators
            for label, indicator in indicator_texts.items():
                extracted_value = find_texts_near_indicator(results, indicator)
                extracted_data.append((label, extracted_value))
            
            # Create a DataFrame to display results
            df = pd.DataFrame(extracted_data, columns=["Field", "Extracted Text"])
            st.table(df)
            
            # Show bounding boxes on the image
            img_with_bboxes = draw_bboxes(image, results)
            st.image(img_with_bboxes, caption='Processed Image with Bounding Boxes', use_column_width=True)

# Function to find texts near indicators
def find_texts_near_indicator(results, indicator_text):
    for result in results:
        if indicator_text in result['text']:
            return result['text']
    return None
