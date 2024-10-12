import cv2
import numpy as np
import pytesseract
from PIL import Image
import streamlit as st
import os

predefined_text_variants = [['نعم انا'],
                            ['اقر بانني', 'اقر باننى'],
                            ['الجنسية'],
                            ['رقم الاثبات'],
                            ['رقم اتصال'],
                            ['الجنسية'],
                            ['رقم الاثبات']]

def find_and_group_sentences_left_of_predefined(sentences, predefined_text_variants, x_range=200, y_range=20):
    """
    Finds and groups sentences to the left of predefined text with the specified conditions.

    Parameters:
    - sentences: List of detected sentence dictionaries with 'text' and 'box'.
    - predefined_text_variants: List of lists, each containing strings of possible text variations.
    - x_range: Maximum X-pixel range allowed to the left of the predefined text.

    Returns:
    - grouped_sentences: List of grouped sentences matching the criteria.
    """
    grouped_sentences = []  # Store grouped sentences for each predefined text
    used_sentences = set()  # Track sentences already used as predefined matches

    for text_group in predefined_text_variants:
        matching_predefined_sentence = None

        # Find the predefined text's sentence box from extracted sentences
        for sentence in sentences:
            if id(sentence) in used_sentences:
                continue

            if any(variant in sentence['text'] for variant in text_group):
                matching_predefined_sentence = sentence
                used_sentences.add(id(sentence))
                break  # Stop after finding the first match for this group

        if not matching_predefined_sentence:
            continue  # Skip if no predefined match found

        indicator_box = matching_predefined_sentence['box']
        indicator_y_top = indicator_box['y']
        indicator_y_bottom = indicator_y_top + indicator_box['h']

        # Collect valid texts to the left of the predefined text
        collected_sentences = []
        collected_text = ""

        for sentence in sentences:
            if id(sentence) in used_sentences:
                continue  # Skip if already used as a predefined match

            box = sentence['box']
            text_y_top = box['y']
            text_y_bottom = text_y_top + box['h']

            # Check if the sentence is on the same line and within x-pixel range to the left
            same_line = (
                (text_y_top >= indicator_y_top - y_range) and
                (text_y_bottom <= indicator_y_bottom + y_range)
            )

            if box['x'] < indicator_box['x'] and (indicator_box['x'] - box['x'] <= x_range) and same_line:
                collected_sentences.append(sentence)
                collected_text += sentence['text'] + " "

        if collected_sentences:
            collected_text = collected_text.strip()  # Remove trailing space
            grouped_sentences.append({
                'predefined_text': text_group,
                'grouped_text': collected_text,
                'collected_sentences': collected_sentences
            })

    return grouped_sentences
    
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

            # Call the function with predefined text variants and extracted sentences
            grouped_sentences = find_and_group_sentences_left_of_predefined(sentences, predefined_text_variants, max_x_threshold=200, y_threshold=10)
            
            # Display the grouped results
            for group in grouped_sentences:
                st.write(f"Predefined Text Group: {group['predefined_text']}")
                st.write(f"Grouped Sentence: {group['grouped_text']}")
                st.write("---")
                        
            # Draw sentences on the cropped image
            processed_image = draw_sentences(cropped_image, sentences)

            # Display the cropped image with sentence-level bounding boxes
            st.image(processed_image, caption=f"Processed Image: {os.path.basename(img_file)}", use_column_width=True)
            
else:
    st.write("No images found in the 'Images' directory.")
