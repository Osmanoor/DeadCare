import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Define fixed bounding boxes (x, y, width, height) for the resized image
FIXED_BBOXES = [
    (50, 100, 300, 150),    # Example coordinates for bounding box 1
    (400, 200, 250, 100),   # Example coordinates for bounding box 2
    (700, 300, 300, 150),   # Example coordinates for bounding box 3
    (50, 500, 400, 200),    # Example coordinates for bounding box 4
    (500, 800, 350, 150),   # Example coordinates for bounding box 5
    (800, 1000, 250, 150)   # Example coordinates for bounding box 6
]

# Detect green horizontal lines
def detect_green_lines(image):
    # Convert the image to HSV (Hue, Saturation, Value) format
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the green color range in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Create a mask for green regions
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours of the green regions
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for horizontal lines based on aspect ratio
    lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if w > 600 and h < 10:  # Adjust thresholds to detect the green lines
            lines.append((x, y, w, h))

    # Sort the lines by their y-coordinate to get top and bottom lines
    lines = sorted(lines, key=lambda line: line[1])
    
    if len(lines) >= 2:
        top_line = lines[0]
        bottom_line = lines[1]
        return top_line, bottom_line
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
    top_line, bottom_line = detect_green_lines(opencv_image)
    
    if top_line and bottom_line:
        # Crop the region between the two green lines
        x, y_top, w, h_top = top_line
        _, y_bottom, _, h_bottom = bottom_line
        cropped_image = opencv_image[y_top + h_top:y_bottom, x:x+w]

        # Resize the cropped image to 990 x 710
        resized_image = cv2.resize(cropped_image, (990, 710))

        # Option to choose the language for OCR (English or Arabic)
        lang_choice = st.selectbox("Choose language for OCR", options=['English', 'Arabic'])
        lang_code = 'eng' if lang_choice == 'English' else 'ara'

        # Button to extract text
        if st.button("Extract Text"):
            extracted_texts = []
            for idx, bbox in enumerate(FIXED_BBOXES):
                extracted_text = extract_text_from_bbox(resized_image, bbox, lang=lang_code)
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
