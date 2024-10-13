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
import json


# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Specify Tesseract OCR config (for Arabic language)
def clean_extracted_text(text):
    # Split the text by lines
    lines = text.strip().split('\n')
    
    # Ensure there are at least two lines to remove the first and last ones
    if len(lines) > 2:
        # Remove the first and last lines
        cleaned_lines = lines[1:-1]
    else:
        # If only one or two lines, return an empty string or handle it differently
        cleaned_lines = []

    # Join the cleaned lines back into a single string
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text

def extract_text(image):
    """Extracts text from an image using Tesseract OCR."""
    config = '--user-words custom_words.txt --oem 3 --psm 6'
    text = pytesseract.image_to_string(image, lang='ara+eng' , config= config)
    return text

def get_gemini_response(ocr_text, prompt):
  """Processes OCR text with Gemini Pro and returns the generated text."""
  model = genai.GenerativeModel('gemini-1.5-flash')
  response = model.generate_content([f"{ocr_text}\n{prompt}"])
  return response.text

def extract_fields(extracted_text):
    """Extracts specific fields from Arabic text using Gemini Pro."""
    prompt = """
    المستند التالي مكتوب باللغة العربية:
    {extracted_text}

    قم باستخراج المعلومات التالية بدقة وفقًا للتعليمات الآتية وقدمها في صيغة JSON:

    1. اسم طالب الخدمة:
       - يمثل اسم شخص بالكامل (قد يكون بالعربي أو بالإنجليزي أو بكلا اللغتين معاً).
       - يجب استخراج الاسم كاملًا، حتى لو كان يتضمن مسافات أو أحرف خاصة (مثل "-" أو "_").
       - الأولوية: أول اسم يظهر في المستند.
       - موقعه المرجح: غالباً يتبع جملة "نعم أنا".
       - لا تتجاهل أي جزء من الاسم عند الاستخراج.

    2. جنسية طالب الخدمة:
       - تمثل اسم دولة أو دولتين بصيغة (دولة بوثيقة دولة أخرى)، مثل "فلسطيني بوثيقة مصرية".
       - الأولوية: أول دولة تظهر في المستند.
       - موقعها المرجح: تأتي غالباً بعد تعريف طالب الخدمة وبعد كلمة "الجنسية".

    3. رقم الهوية الوطنية لطالب الخدمة:
       - رقم مكتوب باللغة الإنجليزية.
       - موقعه المرجح: يأتي بعد اسم طالب الخدمة وجنسيته.
       - غالباً يتبع جملة "رقم الإثبات".

    4. رقم هاتف طالب الخدمة:
       - رقم يتألف من 10 خانات باللغة الإنجليزية ويبدأ بـ 0.
       - موقعه المرجح: يأتي بعد رقم الهوية.
       - غالباً يتبع جملة "رقم اتصال".

    5. اسم المتوفى:
       - يمثل اسم شخص بالكامل (قد يكون بالعربي أو بالإنجليزي أو بكلا اللغتين معاً).
       - يجب استخراج الاسم كاملًا، حتى لو كان يتضمن مسافات أو أحرف خاصة.
       - الأولوية: ثاني اسم يظهر في المستند.
       - موقعه المرجح: غالباً يتبع جملة "أقر بأنني استلمت".
       - لا تتجاهل أي جزء من الاسم عند الاستخراج.

    6. جنسية المتوفى:
       - تمثل دولة أو أكثر بصيغة (دولة بوثيقة دولة أخرى)، مثل "سوري بوثيقة لبنانية".
       - الأولوية: ثاني دولة تظهر في المستند.
       - موقعها المرجح: تأتي بعد اسم المتوفى وبعد كلمة "الجنسية".

    7. رقم الهوية الوطنية للمتوفى:
       - رقم مكتوب باللغة الإنجليزية.
       - موقعه المرجح: يظهر بعد اسم المتوفى وجنسيته.
       - غالباً يتبع جملة "رقم الإثبات"، ويكون ثاني رقم إثبات يظهر في المستند.

    ---

    مثال على التنسيق المطلوب بصيغة JSON:
    {
        "service_requester": {
            "name": "John Doe",
            "nationality": "فلسطيني بوثيقة مصرية",
            "id_number": "1234567890",
            "phone_number": "0123456789"
        },
        "deceased": {
            "name": "Jane Doe",
            "nationality": "سوري بوثيقة لبنانية",
            "id_number": "0987654321"
        }
    }

    ---

     بدون اي اضافات او عناوين فقط JSON الرجاء تقديم الرد بنفس تنسيق JSON أعلاه لضمان سهولة التحليل والعرض في تطبيق Streamlit.
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
st.title("Arabic Document Field Extraction using OCR")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the image if uploaded
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL Image to OpenCV format
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect green lines
    # top_line, bottom_line = detect_gray_lines(image_np)

    if True:
        # Crop the image between the lines
        # x1, y1, w1, h1 = top_line
        # x2, y2, w2, h2 = bottom_line
        cropped_image = image_np[500:image_np.shape[0]-400, :] #image_np[y1+h1:y2, x1:x1+w1]
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_image = cv2.GaussianBlur(cropped_image, (3, 3), 0)

        # Convert the cropped image to PIL format for OCR
        cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # Extract text using Tesseract on the cropped image
        extracted_text = extract_text(cropped_image_pil)
        # st.subheader("Extracted Text:")
        # st.text(extracted_text)

        # Extract fields using Gemini Pro
        if st.button("Extract Fields"):
            with st.spinner("Extracting fields..."):
                extracted_fields = extract_fields(extracted_text)
                # st.subheader("Extracted Fields:")
                data = json.loads(clean_extracted_text(extracted_fields)) 
                st.write("### Service Requester")
                st.table([data["service_requester"]])    
                st.write("### Deceased")
                st.table([data["deceased"]])
                # st.text(extracted_fields)
    else:
        st.warning("Could not detect two lines in the image.")
