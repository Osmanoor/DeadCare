import streamlit as st
from PIL import Image
import pytesseract
import os
from dotenv import load_dotenv
import base64
import io
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Specify Tesseract OCR config (for Arabic language)
custom_config = r'--oem 3 --psm 6 -l ara'

def extract_text(image):
    """Extracts text from an image using Tesseract OCR."""
    text = pytesseract.image_to_string(image, config=custom_config)
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

# Streamlit app
st.title("Arabic Document Field Extraction using OCR and Gemini Pro")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the image if uploaded
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract text using Tesseract
    extracted_text = extract_text(image)
    st.subheader("Extracted Text:")
    st.text(extracted_text)

    # Extract fields using Gemini Pro
    if st.button("Extract Fields"):
        with st.spinner("Extracting fields with Gemini Pro..."):
            extracted_fields = extract_fields(extracted_text)
            st.subheader("Extracted Fields:")
            st.text(extracted_fields)
