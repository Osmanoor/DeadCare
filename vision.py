from PIL import Image
import os
from dotenv import load_dotenv
import base64
import io
import google.generativeai as genai
import cv2
import numpy as np
import json
from google.cloud import vision

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "deadcarereports-abf0d032492d.json"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def clean_extracted_text(text):
    lines = text.strip().split('\n')
    
    if len(lines) > 2:
        cleaned_lines = lines[1:-1]
    else:
        cleaned_lines = []

    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text

def extract_text_vision(base64_image):
    image_bytes = base64.b64decode(base64_image)
    
    image = Image.open(io.BytesIO(image_bytes))
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 

    cropped_image = image_np[500:image_np.shape[0]-400, :] 
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cropped_image = cv2.GaussianBlur(cropped_image, (3, 3), 0)
    cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    text =  texts[0].description if texts else ""
    result = extract_fields(text)
    
    return result

def get_gemini_response(ocr_text, prompt):
  model = genai.GenerativeModel('gemini-1.5-flash')
  response = model.generate_content([f"{ocr_text}\n{prompt}"])
  return response.text

def extract_fields(extracted_text):
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
