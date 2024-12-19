# Dead Care Reports - Advanced Document Processing System

## 📋 Overview
Dead Care Reports is a sophisticated document processing system specifically engineered for handling death certificates and related medical documentation in a bilingual (Arabic/English) environment. The system employs a multi-layered approach combining traditional OCR with advanced AI models to extract, validate, and structure complex document data. Built to handle the unique challenges of Arabic text processing, right-to-left (RTL) script handling, and mixed-language documents, the system provides high accuracy in data extraction while maintaining document processing efficiency.

Key differentiators:
- Specialized handling of Arabic medical terminology
- Intelligent document segmentation using computer vision
- AI-powered contextual understanding
- High-precision data extraction with multi-stage validation
- Real-time processing capabilities
- Scalable architecture supporting both API and UI access

## 🚀 Key Features

### 1. Advanced Document Processing
- **Intelligent Region Detection**
  - Automatic detection of document boundaries using gray line detection
  - Smart cropping of relevant document sections
  - Dynamic adjustment of processing regions based on document layout
  - Custom calibration for different document types

### 2. Multi-Engine OCR Processing
- **Tesseract OCR Integration**
  - Custom word dictionary support for medical terminology
  - Optimized configuration for Arabic text
  - Special handling of mixed script documents
  - Advanced page segmentation modes

- **Google Cloud Vision Integration**
  - High-accuracy text detection
  - Language-agnostic text recognition
  - Robust handling of different fonts and styles
  - Support for degraded document quality

### 3. AI-Powered Text Analysis
- **Gemini AI Integration**
  - Context-aware text understanding
  - Natural language processing for Arabic text
  - Intelligent field extraction and categorization
  - Adaptive learning capabilities

### 4. Document Understanding
- **Layout Analysis**
  - Automatic form field detection
  - Structural document understanding
  - Spatial relationship analysis
  - Dynamic template matching

- **Text Classification**
  - Automatic language detection
  - Content type classification
  - Field type identification
  - Validation rules application

## 💡 Technical Implementation

### Architecture Details
#### 1. Frontend Layer
- **Streamlit Framework**
  ```python
  import streamlit as st
  
  class DocumentProcessor:
      def __init__(self):
          self.supported_formats = ["png", "jpg", "jpeg"]
          self.max_file_size = 10 * 1024 * 1024  # 10MB
          
      def process_document(self, uploaded_file):
          if uploaded_file.type.split('/')[1] in self.supported_formats:
              return self.extract_data(uploaded_file)
  ```

#### 2. Processing Pipeline
- **Document Preprocessing**
  ```python
  def preprocess_image(image):
      # Convert to grayscale
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # Apply Gaussian blur
      blurred = cv2.GaussianBlur(gray, (3, 3), 0)
      # Enhance contrast
      enhanced = cv2.equalizeHist(blurred)
      return enhanced
  ```

- **Region Detection**
  ```python
  def detect_regions(image):
      # Detect horizontal lines
      horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
      detect_horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
      
      # Find contours
      contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
      return contours
  ```

#### 3. OCR Integration
- **Multi-Engine Processing**
  ```python
  class OCRProcessor:
      def __init__(self):
          self.tesseract_config = '--user-words custom_words.txt --oem 3 --psm 6'
          self.vision_client = vision.ImageAnnotatorClient()
          
      def process_text(self, image):
          # Run both OCR engines
          tesseract_result = self.run_tesseract(image)
          vision_result = self.run_vision_api(image)
          
          # Combine and validate results
          return self.merge_results(tesseract_result, vision_result)
  ```

## 🛠️ Technical Stack Details

### Core Components
1. **Image Processing**
   - OpenCV for advanced image manipulation
   - PIL for image handling and conversion
   - Custom image enhancement algorithms

2. **OCR Engines**
   - Tesseract OCR with custom configuration
   - Google Cloud Vision API integration
   - Custom post-processing pipeline

3. **AI/ML Components**
   - Google Gemini AI for text analysis
   - Custom NLP models for Arabic text
   - Machine learning-based validation

4. **Backend Services**
   - FastAPI for RESTful endpoints
   - Async processing capabilities
   - Rate limiting and request validation

## 📱 Usage Examples

### 1. Basic Document Processing
```python
from dead_care import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process single document
result = processor.process_document("path/to/document.jpg")
print(result.extracted_data)
```

### 2. API Integration
```python
import requests

# Process document via API
def process_document_api(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
        
    response = requests.post(
        "http://api.example.com/process_image",
        json={"image_base64": base64_image}
    )
    return response.json()
```

### 3. Batch Processing
```python
async def process_batch(document_list):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for doc in document_list:
            task = asyncio.ensure_future(
                process_single_document(session, doc))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
    return results
```

### 4. Custom Field Extraction
```python
def extract_custom_fields(text_data):
    # Define field patterns
    patterns = {
        'name': r'نعم انا\s+(.*?)(?=\s+الجنسية)',
        'id': r'رقم الاثبات\s+(\d+)',
        'nationality': r'الجنسية\s+(.*?)(?=\s+رقم)'
    }
    
    # Extract fields
    extracted_data = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text_data)
        if match:
            extracted_data[field] = match.group(1)
            
    return extracted_data
```

### 5. Document Validation
```python
class DocumentValidator:
    def validate_document(self, extracted_data):
        required_fields = ['name', 'id_number', 'nationality']
        validation_rules = {
            'id_number': lambda x: len(x) == 10 and x.isdigit(),
            'phone_number': lambda x: x.startswith('0') and len(x) == 10,
            'name': lambda x: len(x.split()) >= 2
        }
        
        return self.apply_validation_rules(extracted_data, validation_rules)
```

These expanded sections provide a more detailed understanding of the system's capabilities and technical implementation. Would you like me to elaborate on any specific aspect further?
