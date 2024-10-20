# import os
# import io
# import base64
# import re
# from PIL import Image, ImageChops
# import numpy as np
# import requests
# from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from fastapi.responses import JSONResponse, Response
# from fastapi.middleware.cors import CORSMiddleware

# # Load environment variable for OCR API key
# OCR_API_KEY = os.getenv('OCR_API_KEY', 'c2a2cadefc88957')
# OCR_API_URL = 'https://api.ocr.space/parse/image'

# TEMPLATE_WIDTH = 1100
# TEMPLATE_HEIGHT = 610.5

# # SVG field mapping with coordinates
# svg_mapped_fields = {
#     'Employee SSN': (0.5, 0.5, 235, 49),
#     'a Employee\'s social security number': (235.5, 1.5, 261, 48),
#     'OMB No .': (497.5, 2.5, 597, 48),
#     'b Employer identification number EIN': (0.5, 49.5, 598, 50),
#     '1 Wages, tips, other compensation': (598.5, 49.5, 249, 49),
#     '2 Federal income tax withheld': (848.5, 49.5, 246, 49),
#     '3 Social security wages': (598.5, 99.5, 249, 49),
#     '4 Social security tax withheld': (848.5, 99.5, 246, 49),
#     '5 Medical wages and tips': (598.5, 149.5, 249, 49),
#     '6 Medicare tax withheld': (848.5, 149.5, 246, 49),
#     '7 Social security tips': (598.5, 197.5, 249, 49),
#     '8 Allocated tips': (848.5, 197.5, 246, 49),
#     '10 Dependent care benefits': (848.5, 247.5, 246, 49),
#     '12a See instructions for box 12': (848.5, 296.5, 246, 47),
#     '12b': (848.5, 343.5, 246, 49),
#     '11 Nonqualified plans': (599.5, 294.5, 249, 49),
#     '12c': (848.5, 392.5, 246, 49),
#     '12d': (848.5, 441.5, 246, 49),
#     'c Employer\'s name, address, and ZIP code': (0.5, 99.5, 598, 147),
#     'd Control number': (0.5, 246.5, 599, 48),
#     'e Employee\'s first name and initial, last name': (0.5, 294.5, 600, 220),
#     'Employer\'s id number': (0.5, 514.5, 322, 46),
#     '16 State wages, tips': (322.5, 514.5, 174, 48),
#     '17 State income tax': (496.5, 514.5, 161, 48),
#     '18 Local wages, tips': (656.5, 514.5, 176, 48),
#     '19 Local income tax': (833.5, 514.5, 159, 48),
# }



#     'Employee SSN':    
# 'a Employee\'s social security number':#     
# 'OMB No .':      'b Employer identification number EIN':     '1 Wages, tips, other compensation': #     '2 Federal income tax withheld': #     '3 Social security wages':
#     '4 Social security tax withheld':
#     '5 Medical wages and tips': 
#     '6 Medicare tax withheld': 
#     '7 Social security tips':,
#     '8 Allocated tips': 
#     '10 Dependent care benefits': 
#     '12a See instructions for box 12': 
#     '12b': 
#     '11 Nonqualified plans': 
#     '12c': 
#     '12d': 
#     'c Employer\'s name, address, and ZIP code':      'd Control number':#     'e Employee\'s first name and initial, last name': #     'Employer\'s id number':#     '16 State wages, tips': 
#     '17 State income tax': 
#     '18 Local wages, tips': 
#     '19 Local income tax': 



# app = FastAPI()

# # CORS settings
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "https://your-next-js-domain.vercel.app"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Utility function to resize the image
# def resize_image(image, target_width, target_height):
#     return image.resize((target_width, target_height), Image.BILINEAR)

# # Utility function to trim whitespace from the image
# def trim_whitespace(image):
#     bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
#     diff = ImageChops.difference(image, bg)
#     diff = ImageChops.add(diff, diff, 2.0, -100)
#     bbox = diff.getbbox()
#     if bbox:
#         return image.crop(bbox)
#     return image

# # Function to send image to OCR API
# def ocr_space_request(cropped_img):
#     buffer = io.BytesIO()
#     cropped_img.save(buffer, format="PNG")
#     img_str = base64.b64encode(buffer.getvalue()).decode()

#     payload = {
#         'apikey': OCR_API_KEY,
#         'base64Image': 'data:image/png;base64,' + img_str,
#         'isTable': True,
#         'OCREngine': 2
#     }

#     try:
#         response = requests.post(OCR_API_URL, data=payload)
#         response.raise_for_status()
#         result = response.json()

#         if 'ParsedResults' in result and result['ParsedResults']:
#             return result['ParsedResults'][0]['ParsedText']
#         else:
#             return ""
#     except requests.exceptions.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"OCR API error: {e}")

# # Cleaning the extracted text
# def clean_extracted_text(field_name, extracted_text):
#     extracted_text = extracted_text.strip()
#     unwanted_phrases = ["For Official Use Only", "VOID"]
#     for phrase in unwanted_phrases:
#         extracted_text = extracted_text.replace(phrase, "")
#     extracted_text = re.sub(re.escape(field_name), '', extracted_text, flags=re.IGNORECASE)

#     money_fields = [
#         "Wages, tips, other compensation", "Federal income tax withheld",
#         "Social security wages", "Social security tax withheld",
#         "Medical wages and tips", "Medicare tax withheld"
#     ]

#     if any(money_field in field_name for money_field in money_fields):
#         money_match = re.search(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', extracted_text)
#         if money_match:
#             extracted_text = money_match.group(0)

#     extracted_text = re.sub(r'\s+', ' ', extracted_text)
#     extracted_text = re.sub(r'[^\w\s\.\,\$\-]', '', extracted_text)

#     return extracted_text.strip()

# # Function to post-process the extracted data
# def post_process_extracted_data(data):
#     processed_data = {}
#     for field, value in data.items():
#         cleaned_value = clean_extracted_text(field, value)
#         processed_data[field] = cleaned_value
#     return processed_data

# # Function to extract text from SVG-mapped fields in the image
# def extract_text_from_svg_fields(image):
#     extracted_data = {}
#     for field_name, coords in svg_mapped_fields.items():
#         x, y, width, height = coords
#         cropped_img = image.crop((x, y, x + width, y + height))
#         extracted_text = ocr_space_request(cropped_img)
#         cleaned_text = clean_extracted_text(field_name, extracted_text)
#         extracted_data[field_name] = cleaned_text
#     return extracted_data

# # Health check endpoint
# @app.get("/")
# def message():
#     return {"message": "Hello, world!"}

# # Endpoint to extract data from uploaded image file
# @app.post("/extract")
# async def extract_w2_data(file: UploadFile = File(...)):
#     try:
#         print(f"Received file: {file.filename}, content type: {file.content_type}")
#         if file.content_type not in ["image/png", "image/jpeg"]:
#             raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG or JPEG image.")

#         image = Image.open(file.file).convert("RGB")
#         trimmed_image = trim_whitespace(image)
#         resized_image = resize_image(trimmed_image, TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
#         extracted_data = extract_text_from_svg_fields(resized_image)
#         cleaned_data = post_process_extracted_data(extracted_data)

#         headers = {
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
#             "Access-Control-Allow-Headers": "*"
#         }

#         return JSONResponse(content={"extracted_data": cleaned_data}, headers=headers)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")

# # Endpoint to handle base64-encoded images
# @app.post("/extract_base64")
# async def extract_w2_data_base64(base64_image: str = Form(...)):
#     try:
#         # Remove the prefix 'data:image/png;base64,' from the base64 string
#         image_data = re.sub('^data:image/.+;base64,', '', base64_image)
        
#         # Decode the base64 string and open the image
#         image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        
#         # Call your image processing functions here
#         trimmed_image = trim_whitespace(image)
#         resized_image = resize_image(trimmed_image, TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
#         extracted_data = extract_text_from_svg_fields(resized_image)
#         cleaned_data = post_process_extracted_data(extracted_data)

#         headers = {
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
#             "Access-Control-Allow-Headers": "*"
#         }

#         return JSONResponse(content={"extracted_data": cleaned_data}, headers=headers)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageChops, UnidentifiedImageError
from pdf2image import convert_from_bytes
import numpy as np
import cv2
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import base64
from pydantic import BaseModel
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import requests
import traceback
import re
from typing import List
import re


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for the uploaded image and current box coordinates
image_data = None
box_coords = None

class BoxCoordinates(BaseModel):
    x: int
    y: int
    width: int
    height: int

# Helper function to trim white space
def trim_whitespace(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    return image.crop(bbox) if bbox else image

# Helper function to detect the largest box
def detect_largest_box(image, threshold=240):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, processed_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]

    if not contours:
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)
import logging

logging.basicConfig(level=logging.INFO)

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}")
    try:
        contents = await file.read()
        logging.info(f"File size: {len(contents)} bytes")
        
        if file.content_type == "application/pdf":
            logging.info("Processing as PDF")
            images = convert_from_bytes(contents)
            logging.info(f"PDF has {len(images)} pages")
            image = images[0]
        else:
            logging.info("Processing as Image")
            image = Image.open(BytesIO(contents)).convert("RGB")

        logging.info("Detecting box coordinates")
        box_coords = detect_largest_box(image)
        
        if box_coords is None:
            raise HTTPException(status_code=400, detail="No detectable box found")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }


        logging.info("Upload successful")
        return JSONResponse(content={"box": box_coords, "image": base64_image}, headers=headers)

    except UnidentifiedImageError:
        logging.error("Unidentified image error")
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Define a Pydantic model for box coordinates
class BoxCoordinates(BaseModel):
    x: int
    y: int
    width: int
    height: int

# Temporary storage for the image
image_data = None

# Helper function to trim whitespace if required
def trim_whitespace(image):
    # Example logic to trim whitespace around an image
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

class CropRequest(BaseModel):
    x: int
    y: int
    width: int
    height: int
    image_base64: str  # Base64 string for the image
    trim: bool = False  # Optional trimming flag

# Function to trim whitespace (if needed)
def trim_whitespace(image: Image.Image) -> Image.Image:
    # Add your logic here to trim whitespace from the image
    return image

@app.post("/crop/")
async def crop_image(request: CropRequest):
    try:
        # Extract data from the request
        x = request.x
        y = request.y
        width = request.width
        height = request.height
        image_base64 = request.image_base64
        trim = request.trim

        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        # Crop the image based on coordinates
        cropped_image = image.crop((x, y, x + width, y + height))

        # Optionally trim whitespace
        if trim:
            cropped_image = trim_whitespace(cropped_image)

        # Convert the cropped image back to base64
        buffer = BytesIO()
        cropped_image.save(buffer, format="PNG")
        cropped_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }

        return JSONResponse(content={"cropped_image": cropped_base64}, headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Path to the template JSON file
TEMPLATES_FILE_PATH = 'w2_templates.json'

# Endpoint to retrieve all templates
@app.get("/templates")
async def get_templates():
    try:
        if os.path.exists(TEMPLATES_FILE_PATH):
            with open(TEMPLATES_FILE_PATH, 'r') as file:
                templates = json.load(file)
                return {"templates": templates}
        else:
            raise HTTPException(status_code=404, detail="Templates file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the templates JSON file")

# Endpoint to retrieve a single template by ID (if you need this feature later)
@app.get("/templates/{template_id}")
async def get_template(template_id: str):
    try:
        if os.path.exists(TEMPLATES_FILE_PATH):
            with open(TEMPLATES_FILE_PATH, 'r') as file:
                templates = json.load(file)
                if template_id in templates:
                    return {"template": templates[template_id]}
                else:
                    raise HTTPException(status_code=404, detail="Template not found")
        else:
            raise HTTPException(status_code=404, detail="Templates file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the templates JSON file")



# # Load environment variable for OCR API key
# OCR_API_KEY = os.getenv('OCR_API_KEY', 'K89973295488957')
# OCR_API_URL = 'https://api.ocr.space/parse/image'

# # Utility function to resize the image
# def resize_image(image, target_width, target_height):
#     return image.resize((target_width, target_height), Image.BILINEAR)

# # Utility function to trim whitespace from the image
# def trim_whitespace(image):
#     bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
#     diff = ImageChops.difference(image, bg)
#     diff = ImageChops.add(diff, diff, 2.0, -100)
#     bbox = diff.getbbox()
#     if bbox:
#         return image.crop(bbox)
#     return image

# # Function to send image to OCR API
# # Function to send image to OCR API
# def ocr_space_request(cropped_img):
#     buffer = io.BytesIO()
#     cropped_img.save(buffer, format="PNG")
#     img_str = base64.b64encode(buffer.getvalue()).decode()

#     payload = {
#         'apikey': OCR_API_KEY,
#         'base64Image': 'data:image/png;base64,' + img_str,
#         'isTable': True,
#         'OCREngine': 2
#     }

#     try:
#         response = requests.post(OCR_API_URL, data=payload)
#         response.raise_for_status()  # This raises an error for bad responses
#         result = response.json()

#         if 'ParsedResults' in result and result['ParsedResults']:
#             return result['ParsedResults'][0]['ParsedText']
#         else:
#             print(f"OCR API response: {result}")  # Log the raw response from the OCR API
#             return ""

#     except requests.exceptions.RequestException as e:
#         print(f"OCR API request failed: {e}")  # More specific error logging
#         raise HTTPException(status_code=500, detail=f"OCR API error: {e}")


# # Function to extract text from the template fields in the image

# def extract_text_from_template(image, fields):
#     extracted_data = {}
#     for field_name, coords in fields.items():
#         try:
#             # Extract coordinates
#             x, y, width, height = coords
#             print(f"Extracting field: {field_name} with coordinates: {x}, {y}, {width}, {height}")
            
#             # Ensure the coordinates are within the image bounds
#             if x + width > image.width or y + height > image.height:
#                 raise ValueError(f"Coordinates for {field_name} exceed image dimensions: {x}, {y}, {width}, {height}")
            
#             # Crop the image
#             cropped_img = image.crop((x, y, x + width, y + height))
#             print(f"Cropped image for {field_name} successfully.")

#             # Extract text using the OCR function
#             extracted_text = ocr_space_request(cropped_img)
#             print(f"Extracted text for {field_name}: {extracted_text}")

#             # Clean extracted text
#             extracted_data[field_name] = extracted_text

#         except Exception as e:
#             print(f"Error extracting {field_name}: {e}")
#             raise HTTPException(status_code=500, detail=f"Error extracting {field_name}: {e}")
    
#     return extracted_data



# @app.post("/extract_text/")
# async def extract_text_from_image_and_template(
#     file: UploadFile = File(...),
#     template: str = Form(...)
# ):
#     try:
#         print(f"Received file: {file.filename}, content type: {file.content_type}")
#         print(f"Template: {template}")

#         if file.content_type not in ["image/png", "image/jpeg"]:
#             raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG or JPEG image.")

#         # Load and process the image
#         try:
#             image = Image.open(file.file).convert("RGB")
#             print("Image loaded successfully")
#         except Exception as e:
#             print(f"Error loading image: {e}")
#             raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

#         try:
#             trimmed_image = trim_whitespace(image)

#             # Convert the template string to a dictionary
#             template_dict = eval(template)
            
#             # Fixing the path to access 'dimensions' inside 'selectedTemplate'
#             dimensions = template_dict['selectedTemplate']['dimensions']
#             print(f"Parsed Template: {template_dict}")  # Print parsed template for debugging
#             print(f"Template Dimensions: {dimensions}")  # Log dimensions

#             # Resize the image to match the template dimensions
#             resized_image = resize_image(trimmed_image, dimensions['width'], dimensions['height'])
#             print(f"Image resized to: {dimensions['width']} x {dimensions['height']}")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error processing template: {e}")

#         # Initialize extracted_data before the extraction process
#         extracted_data = {}

#         # Extract the text from the image based on the template fields
#         try:
#             extracted_data = extract_text_from_template(resized_image, template_dict['selectedTemplate']['fields'])
#             print(f"Text extraction successful. Extracted Data: {extracted_data}")
#         except Exception as e:
#             print(f"Error during text extraction: {e}")
#             raise HTTPException(status_code=500, detail=f"Error extracting text: {e}")

#         # Send back the extracted data
#         headers = {
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
#             "Access-Control-Allow-Headers": "*"
#         }
#         return JSONResponse(content={"extracted_data": extracted_data}, headers=headers)

#     except Exception as e:
#         print(f"General exception: {e}")
#         raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")

import traceback  # To get detailed error messages

# Load environment variable for OCR API key
OCR_API_KEY = os.getenv('OCR_API_KEY', 'K89973295488957')
OCR_API_URL = 'https://api.ocr.space/parse/image'

# Utility function to resize the image
def resize_image(image, target_width, target_height):
    print(f"Resizing image to {target_width}x{target_height}")
    return image.resize((target_width, target_height), Image.BILINEAR)

# Utility function to trim whitespace from the image
def trim_whitespace(image):
    print("Trimming whitespace from the image")
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

# Function to send image to OCR API
def ocr_space_request(cropped_img):
    try:
        print("Preparing image for OCR request")
        buffer = io.BytesIO()
        cropped_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            'apikey': OCR_API_KEY,
            'base64Image': 'data:image/png;base64,' + img_str,
            'isTable': True,
            'OCREngine': 2
        }

        print("Sending OCR request to API")
        response = requests.post(OCR_API_URL, data=payload)
        response.raise_for_status()  # This raises an error for bad responses
        result = response.json()

        if 'ParsedResults' in result and result['ParsedResults']:
            print("OCR response received successfully")
            return result['ParsedResults'][0]['ParsedText']
        else:
            print(f"Unexpected OCR API response: {result}")
            return ""

    except requests.exceptions.RequestException as e:
        print(f"OCR API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR API error: {e}")

# Function to extract text from the template fields in the image
# def extract_text_from_template(image, fields):
#     extracted_data = {}
#     for field_name, coords in fields.items():
#         try:
#             # Extract coordinates
#             x, y, width, height = coords
#             print(f"Extracting field: {field_name} with coordinates: {x}, {y}, {width}, {height}")

#             # Ensure the coordinates are within the image bounds
#             if x + width > image.width or y + height > image.height:
#                 raise ValueError(f"Coordinates for {field_name} exceed image dimensions: {x}, {y}, {width}, {height}")
            
#             # Crop the image
#             cropped_img = image.crop((x, y, x + width, y + height))
#             print(f"Cropped image for {field_name} successfully")

#             # Extract text using the OCR function
#             extracted_text = ocr_space_request(cropped_img)
#             print(f"Extracted text for {field_name}: {extracted_text}")

#             # Clean extracted text
#             extracted_data[field_name] = extracted_text.strip()

#         except Exception as e:
#             print(f"Error extracting {field_name}: {e}")
#             raise HTTPException(status_code=500, detail=f"Error extracting {field_name}: {e}")

#     return extracted_data
import re

# Function to extract text from the template fields in the image
def extract_text_from_template(image, fields):
    extracted_data = {}
    for field_name, coords in fields.items():
        try:
            # Extract coordinates
            x, y, width, height = coords
            print(f"Extracting field: {field_name} with coordinates: {x}, {y}, {width}, {height}")
            
            # Ensure the coordinates are within the image bounds
            if x + width > image.width or y + height > image.height:
                raise ValueError(f"Coordinates for {field_name} exceed image dimensions: {x}, {y}, {width}, {height}")
            
            # Crop the image
            cropped_img = image.crop((x, y, x + width, y + height))
            print(f"Cropped image for {field_name} successfully.")

            # Extract text using the OCR function
            extracted_text = ocr_space_request(cropped_img)
            print(f"Extracted text for {field_name}: {extracted_text}")

            # Clean extracted text
            extracted_data[field_name] = extracted_text

        except Exception as e:
            print(f"Error extracting {field_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error extracting {field_name}: {e}")
    
    return extracted_data

@app.post("/extract_text/")
async def extract_text_from_image_and_template(
    file: UploadFile = File(...),
    template: str = Form(...)
):
    try:
        print(f"Received file: {file.filename}, content type: {file.content_type}")
        print(f"Template received: {template}")

        # Step 1: Validate file type (allow only PNG or JPEG)
        if file.content_type not in ["image/png", "image/jpeg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG or JPEG image.")
        print("File type is valid.")

        # Step 2: Load the image
        try:
            image = Image.open(file.file).convert("RGB")
            print("Image loaded successfully.")
        except Exception as e:
            print(f"Error loading image: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

        # Step 3: Parse the template and extract text based on the coordinates
        try:
            # Convert the template string to a dictionary

            template_dict = eval(template)
            print(f"Template parsed successfully: {template_dict}")

            if 'fields' not in template_dict or 'dimensions' not in template_dict:
                raise HTTPException(status_code=400, detail="Invalid template format: Missing 'fields' or 'dimensions'.")
            fields = template_dict['fields']

            # Step 4: Extract text using the field coordinates
            extracted_data = extract_text_from_template(image, fields)
            print(f"Text extraction successful. Extracted Data: {extracted_data}")

        except Exception as e:
            print(f"Error during template processing or text extraction: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error processing template or extracting text: {e}")

        # Step 5: Return the extracted data
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
        return JSONResponse(content={"extracted_data": extracted_data}, headers=headers)

    except Exception as e:
        print(f"General exception: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")

# @app.post("/extract_text/")
# async def extract_text_from_image_and_template(
#     file: UploadFile = File(...),
#     template: str = Form(...)
# ):
#     try:
#         print(f"Received file: {file.filename}, content type: {file.content_type}")
#         print(f"Template received: {template}")

#         # Step 1: Validate file type (allow only PNG or JPEG)
#         if file.content_type not in ["image/png", "image/jpeg"]:
#             raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG or JPEG image.")
#         print("File type is valid.")

#         # Step 2: Load the image
#         try:
#             image = Image.open(file.file).convert("RGB")
#             print("Image loaded successfully.")
#         except Exception as e:
#             print(f"Error loading image: {traceback.format_exc()}")
#             raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

#         try:

#             # Convert the template string to a dictionary
#             template_dict = eval(template)
#             print(f"Template parsed successfully: {template_dict}")

#             # Ensure 'fields' and 'dimensions' are present in the template
#             if 'fields' not in template_dict or 'dimensions' not in template_dict:
#                 raise HTTPException(status_code=400, detail="Invalid template format: Missing 'fields' or 'dimensions'.")

#             fields = template_dict['fields']
#             dimensions = template_dict['dimensions']

#             # Resize the image to match the template dimensions
#             print(f"Image resized to: {dimensions['width']} x {dimensions['height']}")

#         except Exception as e:
#             print(f"Error processing template: {traceback.format_exc()}")
#             raise HTTPException(status_code=500, detail=f"Error processing template: {e}")

#         # Step 4: Extract text from the image based on the template fields
#         try:
#             print("Extracting text from the image based on template fields")
#             extracted_data = extract_text_from_template(image, fields)
#             print(f"Text extraction successful. Extracted Data: {extracted_data}")
#         except Exception as e:
#             print(f"Error during text extraction: {traceback.format_exc()}")
#             raise HTTPException(status_code=500, detail=f"Error extracting text: {e}")

#         # Step 5: Return the extracted data
#         headers = {
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
#             "Access-Control-Allow-Headers": "*"
#         }
#         return JSONResponse(content={"extracted_data": extracted_data}, headers=headers)

#     except Exception as e:
#         print(f"General exception: {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
# @app.post("/extract_text/")
# async def extract_text_from_image_and_template(
#     file: UploadFile = File(...),
#     template: str = Form(...)
# ):
#     try:
#         print(f"Received file: {file.filename}, content type: {file.content_type}")
#         print(f"Template received: {template}")

#         # Step 1: Validate file type (allow only PNG or JPEG)
#         if file.content_type not in ["image/png", "image/jpeg"]:
#             raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG or JPEG image.")
#         print("File type is valid.")

#         # Step 2: Load the image
#         try:
#             image = Image.open(file.file).convert("RGB")
#             print("Image loaded successfully.")
#         except Exception as e:
#             print(f"Error loading image: {traceback.format_exc()}")
#             raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

#         # Step 3: Parse the template and extract text based on the coordinates
#         try:
#             # Convert the template string to a dictionary
#             template_dict = eval(template)
#             print(f"Template parsed successfully: {template_dict}")

#             # Ensure 'fields' are present in the template
#             if 'selectedTemplate' not in template_dict or 'fields' not in template_dict['selectedTemplate']:
#                 raise HTTPException(status_code=400, detail="Invalid template format: Missing 'selectedTemplate' or 'fields'.")

#             fields = template_dict['selectedTemplate']['fields']

#             # Step 4: Extract text using the field coordinates
#             extracted_data = extract_text_from_template(image, fields)
#             print(f"Text extraction successful. Extracted Data: {extracted_data}")

#         except Exception as e:
#             print(f"Error during template processing or text extraction: {traceback.format_exc()}")
#             raise HTTPException(status_code=500, detail=f"Error processing template or extracting text: {e}")

#         # Step 5: Return the extracted data
#         headers = {
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
#             "Access-Control-Allow-Headers": "*"
#         }
#         return JSONResponse(content={"extracted_data": extracted_data}, headers=headers)

#     except Exception as e:
#         print(f"General exception: {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")




# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)