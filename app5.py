import cv2
import numpy as np
import pytesseract
from PIL import Image
import streamlit as st
import os

def detect_gray_lines(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 600 <= w <= 1000 and 3 <= h <= 30:  # Adjust line width and height ranges accordingly
            horizontal_lines.append((x, y, w, h))

    if len(horizontal_lines) >= 2:
        horizontal_lines = sorted(horizontal_lines, key=lambda line: line[1])
        return horizontal_lines[0], horizontal_lines[1]
    else:
        return None, None

# Grouping detected words into sentences
def group_into_sentences(data, x_threshold=50, y_threshold=20):
    words = []
    for i in range(len(data['text'])):
        if data['text'][i].strip() != '':  # Only consider non-empty words
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text = data['text'][i]
            words.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text})

    # Sort words first by Y (top to bottom) and then by X (left to right)
    # words = sorted(words, key=lambda word: (word['y'], word['x']))

    sentences = []
    current_sentence = []
    current_box = None
    current_text = ""

    # Iterate through sorted word boxes and group them based on proximity
    for word in words:
        if not current_sentence:
            current_sentence.append(word)
            current_box = word
            current_text = word['text']
        else:
            # Check proximity with the last word in the sentence
            last_word = current_sentence[-1]
            if (abs(last_word['x'] - word['w'] - word['x']) < x_threshold and  # Horizontal proximity
                abs(last_word['y'] - word['y']) < y_threshold):  # Vertical proximity
                current_sentence.append(word)
                current_text += " " + word['text']  # Append the word to the sentence
                # Expand the current bounding box to include the new word
                current_box = {
                    'x': min(current_box['x'], word['x']),
                    'y': min(current_box['y'], word['y']),
                    'w': max(current_box['x'] + current_box['w'], word['x'] + word['w']) - min(current_box['x'], word['x']),
                    'h': max(current_box['y'] + current_box['h'], word['y'] + word['h']) - min(current_box['y'], word['y'])
                }
            else:
                # If not close enough, finalize the current sentence and start a new one
                sentences.append({'box': current_box, 'text': current_text})
                current_sentence = [word]
                current_box = word
                current_text = word['text']

    # Append the last sentence if any
    if current_sentence:
        sentences.append({'box': current_box, 'text': current_text})

    return sentences

# Draw sentence boxes and return sentence text
def draw_sentences(image, sentences):
    for sentence in sentences:
        box = sentence['box']
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        # Draw the bounding box for the sentence
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the sentence text
        st.write(f"Detected sentence: {sentence['text']}")
    return image

# Streamlit code to upload image and process
st.title("Tesseract OCR - Group Words into Sentences")

# Load images directly from a directory
image_dir = "images"
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".jpg")]

if len(image_files) > 0:
    for img_file in image_files:
        st.write(f"Processing: {img_file}")
        image = cv2.imread(img_file)

        # Detect gray horizontal lines (assuming you have this function already implemented)
        top_line, bottom_line = detect_gray_lines(image)

        if top_line and bottom_line:
            # Crop the region between the two gray lines
            x, y_top, w, h_top = top_line
            _, y_bottom, _, h_bottom = bottom_line
            cropped_image = image[y_top + h_top:y_bottom, x:x + w]
            cropped_image = cv2.medianBlur(cropped_image, 3) 
            # Convert to PIL image for pytesseract
            cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            # Apply pytesseract to detect text and bounding boxes
            custom_config = f"--user-words custom_words.txt --oem 3 --psm 6"
            data = pytesseract.image_to_data(cropped_pil_image, output_type=pytesseract.Output.DICT, config= custom_config, lang='ara+eng')

            # Group detected words into sentences
            sentences = group_into_sentences(data, x_threshold=20, y_threshold=20)

            # Draw sentences on the cropped image
            processed_image = draw_sentences(cropped_image, sentences)

            # Display the cropped image with sentence-level bounding boxes
            st.image(processed_image, caption=f"Processed Image: {os.path.basename(img_file)}", use_column_width=True)

else:
    st.write("No images found in the 'Images' directory.")
