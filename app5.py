import streamlit as st
import cv2
import numpy as np
import os
import pytesseract
from PIL import Image

# Function to detect gray horizontal lines
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

# Directory containing the images
image_dir = "Images"

# List all jpg images in the folder
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".jpg")]

if len(image_files) > 0:
    for img_file in image_files:
        st.write(f"Processing: {img_file}")
        image = cv2.imread(img_file)

        # Detect gray horizontal lines
        top_line, bottom_line = detect_gray_lines(image)

        if top_line and bottom_line:
            # Crop the region between the two gray lines
            x, y_top, w, h_top = top_line
            _, y_bottom, _, h_bottom = bottom_line
            cropped_image = image[y_top + h_top:y_bottom, x:x + w]

            # Convert to PIL image for pytesseract
            cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            # Apply pytesseract to detect text lines with bounding boxes
            data = pytesseract.image_to_data(cropped_pil_image, output_type=pytesseract.Output.DICT)

            # Draw bounding boxes for lines only
            n_boxes = len(data['level'])
            for i in range(n_boxes):
                if data['level'][i] == 5:  # 'line_num' corresponds to level 5 in Tesseract
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    cropped_image = cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the cropped image with line-level bounding boxes
            st.image(cropped_image, caption=f"Cropped Image with Detected Text Lines: {os.path.basename(img_file)}", use_column_width=True)
else:
    st.write("No images found in the 'Images' directory.")
