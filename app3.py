import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Define fixed bounding boxes (x, y, width, height) for the resized image
FIXED_BBOXES = [
    (400, 140, 420, 70),    # من انا
    (0, 140, 160, 70),   # الجنسية
    (0, 210, 160, 50),   # رقم الاثبات
    (0, 300, 160, 50),   # رقم التواصل
    (400, 350, 420, 70),   # اقرر بانني استلمت
    (0, 350, 160, 70),    # الجنسية
    (0, 430, 160, 50),   # رقم الاثبات
]

# Function to detect gray horizontal lines
def detect_gray_lines(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred_image, 50, 150)

    # st.image(edges, caption="Masked Image", use_column_width=True)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on width/height ratio to find horizontal lines
    horizontal_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(f"{x} - {y} - {w} - {h}")
        aspect_ratio = w / float(h)
        if 600 <= w <= 1000 and 3 <= h <= 30:  # Adjust height range based on line thickness
            horizontal_lines.append((x, y, w, h))

    # Sort lines by their y-coordinate (top to bottom)
    if len(horizontal_lines) >= 2:
        horizontal_lines = sorted(horizontal_lines, key=lambda line: line[1])
        return horizontal_lines[0], horizontal_lines[1]  # Return top and bottom lines
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
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Detect green horizontal lines
    top_line, bottom_line = detect_gray_lines(opencv_image)
    
    if top_line and bottom_line:
        # Crop the region between the two green lines
        x, y_top, w, h_top = top_line
        _, y_bottom, _, h_bottom = bottom_line
        cropped_image = opencv_image[y_top + h_top:y_bottom, x:x+w]

        # Resize the cropped image to 990 x 710
        resized_image = cv2.resize(cropped_image, (990, 710))

        st.image(resized_image, caption="Cropped Image", use_column_width=True)


        # Button to extract text
        if st.button("Extract Text"):
            extracted_texts = []
            for idx, bbox in enumerate(FIXED_BBOXES):
                extracted_text = extract_text_from_bbox(resized_image, bbox, lang='ara+eng')
                extracted_texts.append(extracted_text)
                st.write(f"Text in Bounding Box {idx+1}:")
                st.write(extracted_text)

                # Draw the bounding box on the resized image for debugging
                x, y, w, h = bbox
                cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for debugging

            # Convert the image with bounding boxes back to PIL for display
            debug_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            st.image(debug_image, caption="Image with Bounding Boxes", use_column_width=True)
    else:
        st.error("Could not detect the two green horizontal lines in the image.")
