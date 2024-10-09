import streamlit as st
from PIL import Image, ImageDraw
import pytesseract
import numpy as np

# Function to draw bounding boxes around sentences
def draw_bboxes(image, sentence_results):
    img_draw = image.convert('RGB')
    draw = ImageDraw.Draw(img_draw)

    for result in sentence_results:
        bbox = result['bbox']  # Bounding box coordinates for the sentence
        text = result['text']  # Detected sentence
        # Draw the bounding box
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)
        # Draw the text next to the bounding box
        draw.text((bbox[0], bbox[1]), text, fill="red")  # Draw the detected text at the top-left of the bbox

    return img_draw

# Function to determine if a word is Arabic (RTL) or English (LTR)
def is_arabic(word):
    return True #any("\u0600" <= char <= "\u06FF" for char in word)

# Function to print pytesseract output for debugging
def print_tesseract_data(data):
    print("Tesseract Data:")
    for i in range(len(data['text'])):
        print(f"Word: {data['text'][i]}, X: {data['left'][i]}, Y: {data['top'][i]}, W: {data['width'][i]}, H: {data['height'][i]}")

# Function to group words into sentences based on Y-threshold and X-threshold
def get_sentences_with_bboxes(image, x_threshold=50, y_threshold=20):
    # Get word-level data with bbox coordinates
    data = pytesseract.image_to_data(image, lang='ara+eng', output_type=pytesseract.Output.DICT)
    
    # Debug: Print pytesseract data to check word extraction
    print_tesseract_data(data)
    
    sentences = []
    sentence_bbox = None
    sentence_text = ''
    
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            is_rtl = is_arabic(word)  # Check if the current word is Arabic (RTL)

            # Check if we should start a new sentence
            if sentence_bbox is None:
                # Initialize a new sentence and bbox
                sentence_bbox = [x, y, x + w, y + h]
                sentence_text = word
            else:
                # Calculate Y-distance and X-distance
                y_distance = abs(y - sentence_bbox[1])  # Y distance between current word and the top of the sentence
                if is_rtl:
                    x_distance = sentence_bbox[0] - (x + w)  # For RTL: Distance from current word to left of sentence
                else:
                    x_distance = x - sentence_bbox[2]  # For LTR: Distance from current word to right of sentence
                
                # Group words if they are within the Y-threshold and X-threshold
                if y_distance <= y_threshold and x_distance <= x_threshold:
                    # Update bbox to include the new word
                    sentence_bbox[0] = min(sentence_bbox[0], x)  # Update left edge for RTL
                    sentence_bbox[2] = max(sentence_bbox[2], x + w)  # Update right edge for LTR
                    sentence_bbox[3] = max(sentence_bbox[3], y + h)  # Update bottom edge
                    sentence_text += ' ' + word  # Append word to sentence
                else:
                    # Save the previous sentence before starting a new one
                    sentences.append({'text': sentence_text, 'bbox': sentence_bbox})
                    # Start a new sentence
                    sentence_bbox = [x, y, x + w, y + h]
                    sentence_text = word
            
            # If this is the last word, save the current sentence
            if i == len(data['text']) - 1:
                sentences.append({'text': sentence_text, 'bbox': sentence_bbox})
    
    return sentences

# Streamlit app
st.title("Text Extraction with Sentence-Level Bounding Boxes (RTL and LTR)")

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
            # Get sentences with bounding boxes
            sentence_results = get_sentences_with_bboxes(image, x_threshold=50, y_threshold=20)

            # Display the extracted sentences
            st.subheader("Extracted Sentences with BBoxes:")
            for result in sentence_results:
                st.text(f"Sentence: {result['text']}, BBox: {result['bbox']}")
            
            # Draw bounding boxes on the image
            img_with_bboxes = draw_bboxes(image, sentence_results)
    
            # Show image with bounding boxes
            st.image(img_with_bboxes, caption='Processed Image with Bounding Boxes', use_column_width=True)
