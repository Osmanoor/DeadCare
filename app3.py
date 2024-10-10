import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Define fixed bounding boxes (x, y, width, height) for an image with height 700 px
FIXED_BBOXES = [
    (400, 140, 420, 70),    # من انا
    (0, 140, 160, 70),   # الجنسية
    (0, 210, 160, 50),   # رقم الاثبات
    (0, 300, 160, 50),   # رقم التواصل
    (400, 350, 420, 70),   # اقرر بانني استلمت
    (0, 350, 160, 70),    # الجنسية
    (0, 430, 160, 50),   # رقم الاثبات
]

DEFAULT_HEIGHT = 700  # The original height that the fixed bounding boxes are based on

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

# Function to dynamically adjust bounding boxes based on image height
def adjust_bboxes(bboxes, current_height):
    scale_factor = current_height / DEFAULT_HEIGHT
    adjusted_bboxes = [(x, int(y * scale_factor), w, h) for x, y, w, h in bboxes]
    return adjusted_bboxes

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

    top_line, bottom_line = detect_gray_lines(opencv_image)
    
    if top_line and bottom_line:
        x, y_top, w, h_top = top_line
        _, y_bottom, _, h_bottom = bottom_line
        cropped_image = opencv_image[y_top + h_top:y_bottom, x:x+w]
        
        # Get the actual height of the cropped image
        cropped_height = cropped_image.shape[0]

        # Adjust the bounding boxes based on the current height of the cropped image
        adjusted_bboxes = adjust_bboxes(FIXED_BBOXES, cropped_height)

        st.image(cropped_image, caption="Cropped Image", use_column_width=True)

        if st.button("Extract Text"):
            extracted_texts = []
            for idx, bbox in enumerate(adjusted_bboxes):
                extracted_text = extract_text_from_bbox(cropped_image, bbox, lang='ara+eng')
                extracted_texts.append(extracted_text)
                st.write(f"Text in Bounding Box {idx+1}:")
                st.write(extracted_text)

                # Draw the bounding box on the cropped image for debugging
                x, y, w, h = bbox
                cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            debug_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            st.image(debug_image, caption="Image with Bounding Boxes", use_column_width=True)
    else:
        st.error("Could not detect the two green horizontal lines in the image.")
