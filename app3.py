import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Define fixed bounding boxes (x, y, width, height) for the resized image
FIXED_BBOXES = [
    (400, 140, 420, 70),    # من انا
    (0, 140, 160, 70),      # الجنسية
    (0, 210, 160, 50),      # رقم الاثبات
    (0, 300, 160, 50),      # رقم التواصل
    (400, 370, 410, 70),    # اقرر بانني استلمت
    (0, 360, 160, 70),      # الجنسية
    (0, 275, 160, 70),      # رقم الاثبات
]

# Function to adjust bounding boxes for the last three items
def adjust_bboxes(bboxes, image_height):
    adjusted_bboxes = []
    
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        if i < len(bboxes) - 3:
            # For the first four bounding boxes, keep them unchanged
            adjusted_bboxes.append((x, y, w, h))
        else:
            # For the last three bounding boxes, make their y relative to the bottom
            new_y = image_height - y
            adjusted_bboxes.append((x, new_y, w, h))
    
    return adjusted_bboxes

# Function to detect gray horizontal lines
def detect_gray_lines(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 600 <= w <= 1000 and 3 <= h <= 30:
            horizontal_lines.append((x, y, w, h))

    if len(horizontal_lines) >= 2:
        horizontal_lines = sorted(horizontal_lines, key=lambda line: line[1])
        return horizontal_lines[0], horizontal_lines[1]
    else:
        return None, None

# Extract text from a bounding box
def extract_text_from_bbox(image, bbox, lang='eng'):
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w]
    pil_image = Image.fromarray(cropped_image)
    extracted_text = pytesseract.image_to_string(pil_image, lang=lang)
    return extracted_text

# Streamlit App
st.title("Text Extraction from Specific Areas")
st.write("Upload a document image, detect lines, and extract text from predefined bounding boxes.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Detect green horizontal lines
    top_line, bottom_line = detect_gray_lines(opencv_image)
    
    if top_line and bottom_line:
        # Crop the region between the two green lines
        x, y_top, w, h_top = top_line
        _, y_bottom, _, h_bottom = bottom_line
        cropped_image = opencv_image[y_top + h_top:y_bottom, x:x+w]
        cropped_height = cropped_image.shape[0]
        cropped_width = cropped_image.shape[1]
        # Resize the cropped image to 990 x 710
        resized_image = cv2.resize(cropped_image, (cropped_width, 710))

        st.image(resized_image, caption="Cropped Image", use_column_width=True)

        # Adjust the bounding boxes for the resized image
        adjusted_bboxes = adjust_bboxes(FIXED_BBOXES, 710)

        # Button to extract text
        if st.button("Extract Text"):
            extracted_texts = []
            for idx, bbox in enumerate(adjusted_bboxes):
                extracted_text = extract_text_from_bbox(resized_image, bbox, lang='ara+eng')
                extracted_texts.append(extracted_text)
                st.write(f"Text in Bounding Box {idx+1}:")
                st.write(extracted_text)

                # Draw the bounding box on the resized image for debugging
                x, y, w, h = bbox
                cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for debugging

            debug_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            st.image(debug_image, caption="Image with Bounding Boxes", use_column_width=True)
    else:
        st.error("Could not detect the two green horizontal lines in the image.")
