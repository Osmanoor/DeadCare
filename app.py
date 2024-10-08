import streamlit as st
from PIL import Image, ImageDraw
import pytesseract
import numpy as np
import pandas as pd

# Function to draw bounding boxes around detected text
def draw_bboxes(image, results):
    # Convert image to RGB (if not already in RGB mode)
    img_draw = image.convert('RGB')
    draw = ImageDraw.Draw(img_draw)

    for result in results:
        bbox = result['bbox']  # Bounding box coordinates
        text = result['text']  # Detected text
        # Draw the bounding box
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)
        # Optionally, draw the text next to the bounding box
        draw.text((bbox[0], bbox[1]), text, fill="red")  # Draw the detected text at the top-left of the bbox

    return img_draw

# Function to process text and bounding boxes using PyTesseract
def get_text_and_bboxes(image):
    # Use PyTesseract to get bounding boxes and text
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang="ara+eng")
    
    results = []
    for i in range(len(data['text'])):
        if data['text'][i].strip():  # If text is not empty
            # Collect bounding box and text
            bbox = [data['left'][i], data['top'][i], data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]]
            results.append({"bbox": bbox, "text": data['text'][i]})
    
    return results

# Function to find texts to the left of indicators
def find_texts_to_left(results, indicator_text_list, x_range=200):
    indicator_box = None
    for result in results:
        text = result['text']  # Extracted text
        bbox = result['bbox']  # Bounding box

        # Find the indicator bounding box if any of the possible indicators are found
        if any(indicator_text in text for indicator_text in indicator_text_list):
            indicator_box = bbox
            break

    if indicator_box:
        # Get the Y range of the indicator (same line check)
        indicator_y_top = indicator_box[1]  # Top y-coordinate
        indicator_y_bottom = indicator_box[3]  # Bottom y-coordinate

        # Store all valid texts within x-range and same line
        valid_texts = []
        
        for result in results:
            bbox = result['bbox']
            text = result['text']

            # Calculate the Y range of the current text (to check if it is on the same line)
            text_y_top = bbox[1]
            text_y_bottom = bbox[3]

            # Check if text is on the same horizontal line (Y range overlaps with indicator)
            same_line = (text_y_top >= indicator_y_top - 20) and (text_y_bottom <= indicator_y_bottom + 30)

            # Check if the text is within the x-pixel range to the left of the indicator
            if bbox[0] < indicator_box[0] and (indicator_box[0] - bbox[0] <= x_range) and same_line:
                valid_texts.append((bbox[0], text))  # Store x-coordinate and the text

        # Sort the valid texts by their x-coordinates (from left to right)
        sorted_texts = sorted(valid_texts, key=lambda x: x[0], reverse=True)

        # Combine all sorted texts into a single string
        final_text = ' '.join([text for _, text in sorted_texts])
        return final_text

    return None

# Streamlit app
st.title("Arabic and English Text Extraction with PyTesseract")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the image if uploaded
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract text button
    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            # Get bounding boxes and text using PyTesseract
            results = get_text_and_bboxes(image)

            # Define the x-range for searching (in pixels)
            x_range = 300

            # List of possible indicator text variations
            name_1_indicators = ["نعم انا"]
            nationality_1_indicators = ["الجنسية"]
            name_2_indicators = ["اقر بانني استلمت", "اقر باننى استلمت"]
            nationality_2_indicators = ["الجنسية"]

            # Find the specific texts using the function with indicator lists
            name_1 = find_texts_to_left(results, name_1_indicators, x_range)
            nationality_1 = find_texts_to_left(results, nationality_1_indicators, x_range)
            name_2 = find_texts_to_left(results, name_2_indicators, x_range)
            nationality_2 = find_texts_to_left(results, nationality_2_indicators, x_range)

            # Display the extracted specific sentences
            st.subheader("Extracted Information:")
            data = {
                "Information": [
                    f"Name (left of {name_1_indicators})",
                    f"Nationality (left of {nationality_1_indicators})",
                    f"Name (left of {name_2_indicators})",
                    f"Nationality (left of {nationality_2_indicators})"
                ],
                "Extracted Text": [
                    name_1,
                    nationality_1,
                    name_2,
                    nationality_2
                ]
            }
            
            # Convert the data into a pandas DataFrame
            df = pd.DataFrame(data)
            
            # Display the table using st.table
            st.subheader("Extracted Information:")
            st.table(df)

            # Draw bounding boxes on the image
            img_with_bboxes = draw_bboxes(image, results)
    
            # Show image with bounding boxes
            st.image(img_with_bboxes, caption='Processed Image with Bounding Boxes', use_column_width=True)
    
            # Display all recognized text
            for result in results:
                st.text(result['text'])
