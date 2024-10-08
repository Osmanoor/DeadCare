from google.cloud import vision
import io
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "deadcarereports-abf0d032492d.json"


client = vision.ImageAnnotatorClient()

# Load image from file
content = io.open("images/1.jpg", "rb").read()
image = vision.Image(content=content)
# Perform text detection
response = client.text_detection(image=image)
texts = response.text_annotations
print(texts)