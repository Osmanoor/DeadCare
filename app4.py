import streamlit as st
from PIL import Image
import pytesseract
import os
from dotenv import load_dotenv
import base64
import io
import google.generativeai as genai
import cv2
import numpy as np

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Specify Tesseract OCR config (for Arabic language)


def extract_text(image):
    """Extracts text from an image using Tesseract OCR."""
    text = pytesseract.image_to_string(image, lang='ara+eng')
    return text

def get_gemini_response(ocr_text, prompt):
  """Processes OCR text with Gemini Pro and returns the generated text."""
  model = genai.GenerativeModel('gemini-pro')
  response = model.generate_content([f"{ocr_text}\n{prompt}"])
  return response.text

def extract_fields(extracted_text):
    """Extracts specific fields from Arabic text using Gemini Pro."""
    prompt = f"""
    المستند التالي مكتوب باللغة العربية:
    {extracted_text}

    من فضلك، قم باستخراج البيانات التالية من النص:
    - اسم طالب الخدمة
    - جنسية طالب الخدمة
    - رقم الهوية الوطنية لطالب الخدمة
    - رقم هاتف طالب الخدمة
    - اسم المتوفي
    - جنسية المتوفي
    - رقم الهوية الوطنية للمتوفي

    يرجى تقديم البيانات المستخرجة بالتنسيق التالي:
    اسم طالب الخدمة: [الاسم]
    جنسية طالب الخدمة: [الجنسية]
    رقم الهوية الوطنية لطالب الخدمة: [رقم الهوية]
    رقم هاتف طالب الخدمة: [رقم الهاتف]
    اسم المتوفي: [الاسم]
    جنسية المتوفي: [الجنسية]
    رقم الهوية الوطنية للمتوفي: [رقم الهوية]
    """
    return get_gemini_response(extracted_text, prompt)

def detect_gray_lines(image):
    """Detects two horizontal green lines in an image and returns their coordinates."""
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on width/height ratio to find horizontal lines
    horizontal_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 600 <= w <= 1000 and 3 <= h <= 30:  # Adjust height range based on line thickness
            horizontal_lines.append((x, y, w, h))

    # Sort lines by their y-coordinate (top to bottom)
    if len(horizontal_lines) >= 2:
        horizontal_lines = sorted(horizontal_lines, key=lambda line: line[1])
        return horizontal_lines[0], horizontal_lines[1]  # Return top and bottom lines
    else:
        return None, None

# Streamlit app
st.title("Arabic Document Field Extraction using OCR and Gemini Pro")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the image if uploaded
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL Image to OpenCV format
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect green lines
    top_line, bottom_line = detect_gray_lines(image_np)

    if top_line is not None and bottom_line is not None:
        # Crop the image between the lines
        x1, y1, w1, h1 = top_line
        x2, y2, w2, h2 = bottom_line
        cropped_image = image_np[y1+h1:y2, x1:x1+w1]
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)
        cropped_image = cv2.medianBlur(cropped_image, 3) 

        
        # Convert the cropped image to PIL format for OCR
        cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # Extract text using Tesseract on the cropped image
        extracted_text = extract_text(cropped_image_pil)
        st.subheader("Extracted Text:")
        st.text(extracted_text)

        # Extract fields using Gemini Pro
        if st.button("Extract Fields"):
            with st.spinner("Extracting fields with Gemini Pro..."):
                extracted_fields = extract_fields(extracted_text)
                st.subheader("Extracted Fields:")
                st.text(extracted_fields)
    else:
        st.warning("Could not detect two green lines in the image.")
