import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Define fixed bounding boxes (x, y, width, height)
FIXED_BBOXES = [
    (50, 100, 300, 150),    # Example coordinates for bounding box 1
    (400, 200, 250, 100),   # Example coordinates for bounding box 2
    (700, 300, 300, 150),   # Example coordinates for bounding box 3
    (50, 500, 400, 200),    # Example coordinates for bounding box 4
    (500, 800, 350, 150),   # Example coordinates for bounding box 5
    (800, 1000, 250, 150)   # Example coordinates for bounding box 6
]

# Define a function to extract text from the specified bounding box
def extract_text_from_bbox(image, bbox, lang='eng'):
    x, y, w, h = bbox
    # Crop the image using the bounding box coordinates
    cropped_image = image[y:y+h, x:x+w]
    # Convert cropped image to PIL format for pytesseract
    pil_image = Image.fromarray(cropped_image)
    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(pil_image, lang=lang)
    return extracted_text

# Streamlit App
st.title("Text Extraction from Fixed Bboxes")
st.write("Upload an image and extract text from 6 predefined bounding boxes.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Resize the image to 1126 x 1600
    resized_image = cv2.resize(opencv_image, (1126, 1600))
  
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
