from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import google.generativeai as genai
import cv2
import numpy as np
from google.cloud import vision
from io import BytesIO
import os
import json

app = FastAPI()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "conf/deadcarereports.json"
genai.configure(api_key="AIzaSyAGTRCscX-fLYdBJnj_kbODyOW6ljKgD7g")

class ImageData(BaseModel):
    image_base64: str


def extract_text_vision(base64_image):
    image_bytes = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_bytes))
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cropped_image = image_np[500:image_np.shape[0] - 400, :]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cropped_image = cv2.GaussianBlur(cropped_image, (3, 3), 0)
    img_byte_arr = io.BytesIO()
    Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)).save(img_byte_arr, format='PNG')

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img_byte_arr.getvalue())
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # List to store text with its bounding box coordinates
    text_with_boxes = []

    # Extract the text and its bounding box coordinates
    for text in texts:
        # Get the bounding box (coordinates of the vertices)
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        y_coordinates = [vertex.y for vertex in text.bounding_poly.vertices]
        if any((y > 40 and y < 900) for y in y_coordinates):
            # Add text and bounding box info to the list
            text_with_boxes.append({
                "text": text.description,
                "bounding_box": vertices
            })

    # Sort texts first by y-coordinate (top to bottom)
    text_with_boxes.sort(key=lambda x: x['bounding_box'][0][1])

    # Group texts into rows based on their y-coordinate proximity
    rows = []
    current_row = []
    threshold = 10  # Set a threshold for determining when a new row starts (adjust as needed)

    for item in text_with_boxes:
        if not current_row:
            current_row.append(item)
        else:
            # Check if the current text is in the same row (within the threshold)
            last_y = current_row[-1]['bounding_box'][0][1]
            current_y = item['bounding_box'][0][1]

            if abs(current_y - last_y) < threshold:
                current_row.append(item)
            else:
                # Sort the current row by x-coordinate (Right to Left for RTL text)
                current_row.sort(key=lambda x: min(point[0] for point in x['bounding_box']), reverse=True)
                rows.append(current_row)
                current_row = [item]

    # Append the last row
    if current_row:
        # Sort the current row by x-coordinate (Right to Left for RTL text)
        current_row.sort(key=lambda x: min(point[0] for point in x['bounding_box']), reverse=True)
        rows.append(current_row)

    # Combine the texts into final output, preserving row-wise sorting for RTL text
    final_text = "\n".join(" ".join([item['text'] for item in row]) for row in rows)

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    result = extract_fields(final_text)

    return json.loads(result)


def extract_fields(extracted_text):
    prompt = """
المستند في الأعلى مكتوب باللغة العربية، قم باستخراج المعلومات التالية بدقة وفقًا للتعليمات الآتية وقدمها في صيغة JSON:

1. اسم طالب الخدمة:
   - يمثل اسم شخص بالكامل (قد يكون بالعربي أو بالإنجليزي أو بكلا اللغتين معاً).
   - يجب استخراج الاسم كاملًا، حتى لو كان يتضمن مسافات أو أحرف خاصة (مثل "-" أو "_").
   - الأولوية: أول اسم يظهر في المستند.
   - موقعه المرجح: غالباً يتبع جملة "نعم أنا".
   - لا تتجاهل أي جزء من الاسم عند الاستخراج.

2. رقم الهوية الوطنية لطالب الخدمة:
   - رقم مكتوب باللغة الإنجليزية.
   - موقعه المرجح: يأتي بعد اسم طالب الخدمة وجنسيته.
   - غالباً يتبع جملة "رقم الإثبات".

3. رقم هاتف طالب الخدمة:
   - رقم يتألف من 10 خانات باللغة الإنجليزية ويبدأ بـ 0.
   - موقعه المرجح: يأتي بعد رقم الهوية.
   - غالباً يتبع جملة "رقم اتصال".

4. اسم المتوفى:
   - يمثل اسم شخص بالكامل (قد يكون بالعربي أو بالإنجليزي أو بكلا اللغتين معاً).
   - يجب استخراج الاسم كاملًا، حتى لو كان يتضمن مسافات أو أحرف خاصة.
   - الأولوية: ثاني اسم يظهر في المستند.
   - موقعه المرجح: غالباً يتبع جملة "أقر بأنني استلمت".
   - لا تتجاهل أي جزء من الاسم عند الاستخراج.

5. جنسية المتوفى:
   - تمثل دولة أو أكثر بصيغة (دولة بوثيقة دولة أخرى)، مثل "سوري بوثيقة لبنانية".
   - الأولوية: ثاني دولة تظهر في المستند.
   - موقعها المرجح: تأتي بعد اسم المتوفى وبعد كلمة "الجنسية".
   - اعرضها برمز الدولة حسب الرموز الموجودة في نظام أودو 13 نسخة المجتمع، مثلا (سعودي) اكتبها SA

6. رقم الهوية الوطنية للمتوفى:
   - رقم مكتوب باللغة الإنجليزية.
   - موقعه المرجح: يظهر بعد اسم المتوفى وجنسيته.
   - غالباً يتبع جملة "رقم الإثبات"، ويكون ثاني رقم إثبات يظهر في المستند.

7. عمر المتوفى:
   - رقم مكتوب باللغة الانجليزية. 
   - موقعه المرجح: يظهر أسفل اسم المتوفى.
   - غالباً يتبع كلمة (العمر).

8. جنس المتوفى:
   - تقوم بمعرفته بناء على معرفتك بالأسماء الخاصة بالرجال والنساء. 
   - القيم المحتملة للجنس:
   1- ذكر ويتم الرمز له برقم (1) بدون أقواس، ويكون لكل متوفى اسمه يوحي بأنه ذكر ويكون العمر أكبر من 13 سنة
   2- أنثى ويتم الرمز له برقم (2) بدون أقواس، ويكون لكل متوفى اسمه يوحي بأنه أنثى ويكون العمر أكبر من 13 سنة
   3- طفل ويتم الرمز له برقم (3) بدون أقواس، ويكون لكل متوفى اسمه يوحي بأنه ذكر ويكون العمر أصغر من 14 سنة ويمكن أن يكون 0 سنة يعني عمره أقل من سنة
   4- طفلة ويتم الرمز له برقم (4) بدون أقواس، ويكون لكل متوفى اسمه يوحي بأنه أنثى ويكون العمر أصغر من 14 سنة ويمكن أن يكون 0 سنة يعني عمره أقل من سنة
   
9. اسم المستشفى:
   - المستشفى الذي تمت فيه الوفاة.
   - موقعه المرجح: يظهر في أعلى النص.
   - غالباً يبدأ بكلمة مستشفى، ولكن يمكن أن يكون (مجمع) أو (مدينة الملك سعود الطبية).

---

مثال على التنسيق المطلوب بصيغة JSON:
{
    "service_requester": {
        "name": "John Doe",
        "id_number": "1234567890",
        "phone_number": "0123456789"
    },
    "dead_data": {
        "name": "Jane Doe",
        "nationality": "SA",
        "id_number": "0987654321"
        "gender": "male"
    }
}

---

بدون اي اضافات او عناوين فقط JSON الرجاء تقديم الرد بنفس تنسيق JSON أعلاه لضمان سهولة التحليل والعرض في تطبيق Streamlit.
وأرسل الرد بشكل طبيعي دون إضافة ```json
"""
    return get_gemini_response(extracted_text, prompt)

def get_gemini_response(ocr_text, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([f"{ocr_text}\n{prompt}"])
    return response.text

@app.post("/process_image")
def process_image(image_data: ImageData):
    try:
        result = extract_text_vision(image_data.image_base64)
        return result
    except Exception as e:
        return {"message": f"Error decoding base64: {e}"}
