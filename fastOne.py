import os
import io
import base64
import re
from PIL import Image, ImageChops
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# Load environment variable for OCR API key
OCR_API_KEY = os.getenv('OCR_API_KEY', 'c2a2cadefc88957')
OCR_API_URL = 'https://api.ocr.space/parse/image'

TEMPLATE_WIDTH = 1100
TEMPLATE_HEIGHT = 701

# SVG field mapping with coordinates
svg_mapped_fields = {
    'Employee SSN': (0.5, 0.5, 235, 49),
    'a Employee\'s social security number': (235.5, 1.5, 261, 48),
    'OMB No .': (497.5, 2.5, 597, 48),
    'b Employer identification number EIN': (0.5, 49.5, 598, 50),
    '1 Wages, tips, other compensation': (598.5, 49.5, 249, 49),
    '2 Federal income tax withheld': (848.5, 49.5, 246, 49),
    '3 Social security wages': (598.5, 99.5, 249, 49),
    '4 Social security tax withheld': (848.5, 99.5, 246, 49),
    '5 Medical wages and tips': (598.5, 149.5, 249, 49),
    '6 Medicare tax withheld': (848.5, 149.5, 246, 49),
    '7 Social security tips': (598.5, 197.5, 249, 49),
    '8 Allocated tips': (848.5, 197.5, 246, 49),
    '10 Dependent care benefits': (848.5, 247.5, 246, 49),
    '12a See instructions for box 12': (848.5, 296.5, 246, 47),
    '12b': (848.5, 343.5, 246, 49),
    '11 Nonqualified plans': (599.5, 294.5, 249, 49),
    '12c': (848.5, 392.5, 246, 49),
    '12d': (848.5, 441.5, 246, 49),
    'c Employer\'s name, address, and ZIP code': (0.5, 99.5, 598, 147),
    'd Control number': (0.5, 246.5, 599, 48),
    'e Employee\'s first name and initial, last name': (0.5, 294.5, 600, 220),
    'Employer\'s id number': (0.5, 514.5, 322, 46),
    '16 State wages, tips': (322.5, 514.5, 174, 48),
    '17 State income tax': (496.5, 514.5, 161, 48),
    '18 Local wages, tips': (656.5, 514.5, 176, 48),
    '19 Local income tax': (833.5, 514.5, 159, 48),
}

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-next-js-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility function to resize the image
def resize_image(image, target_width, target_height):
    return image.resize((target_width, target_height), Image.BILINEAR)

# Utility function to trim whitespace from the image
def trim_whitespace(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

# Function to send image to OCR API
def ocr_space_request(cropped_img):
    buffer = io.BytesIO()
    cropped_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    payload = {
        'apikey': OCR_API_KEY,
        'base64Image': 'data:image/png;base64,' + img_str,
        'isTable': True,
        'OCREngine': 2
    }

    try:
        response = requests.post(OCR_API_URL, data=payload)
        response.raise_for_status()
        result = response.json()

        if 'ParsedResults' in result and result['ParsedResults']:
            return result['ParsedResults'][0]['ParsedText']
        else:
            return ""
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OCR API error: {e}")

# Cleaning the extracted text
def clean_extracted_text(field_name, extracted_text):
    extracted_text = extracted_text.strip()
    unwanted_phrases = ["For Official Use Only", "VOID"]
    for phrase in unwanted_phrases:
        extracted_text = extracted_text.replace(phrase, "")
    extracted_text = re.sub(re.escape(field_name), '', extracted_text, flags=re.IGNORECASE)

    money_fields = [
        "Wages, tips, other compensation", "Federal income tax withheld",
        "Social security wages", "Social security tax withheld",
        "Medical wages and tips", "Medicare tax withheld"
    ]

    if any(money_field in field_name for money_field in money_fields):
        money_match = re.search(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', extracted_text)
        if money_match:
            extracted_text = money_match.group(0)

    extracted_text = re.sub(r'\s+', ' ', extracted_text)
    extracted_text = re.sub(r'[^\w\s\.\,\$\-]', '', extracted_text)

    return extracted_text.strip()

# Function to post-process the extracted data
def post_process_extracted_data(data):
    processed_data = {}
    for field, value in data.items():
        cleaned_value = clean_extracted_text(field, value)
        processed_data[field] = cleaned_value
    return processed_data

# Function to extract text from SVG-mapped fields in the image
def extract_text_from_svg_fields(image):
    extracted_data = {}
    for field_name, coords in svg_mapped_fields.items():
        x, y, width, height = coords
        cropped_img = image.crop((x, y, x + width, y + height))
        extracted_text = ocr_space_request(cropped_img)
        cleaned_text = clean_extracted_text(field_name, extracted_text)
        extracted_data[field_name] = cleaned_text
    return extracted_data

# Health check endpoint
@app.get("/")
def message():
    return {"message": "Hello, world!"}

# Endpoint to extract data from uploaded image file
@app.post("/extract")
async def extract_w2_data(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}, content type: {file.content_type}")
        if file.content_type not in ["image/png", "image/jpeg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG or JPEG image.")

        image = Image.open(file.file).convert("RGB")
        trimmed_image = trim_whitespace(image)
        resized_image = resize_image(trimmed_image, TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
        extracted_data = extract_text_from_svg_fields(resized_image)
        cleaned_data = post_process_extracted_data(extracted_data)

        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }

        return JSONResponse(content={"extracted_data": cleaned_data}, headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")

# Endpoint to handle base64-encoded images
@app.post("/extract_base64")
async def extract_w2_data_base64(base64_image: str = Form(...)):
    try:
        # Remove the prefix 'data:image/png;base64,' from the base64 string
        image_data = re.sub('^data:image/.+;base64,', '', base64_image)
        
        # Decode the base64 string and open the image
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        
        # Call your image processing functions here
        trimmed_image = trim_whitespace(image)
        resized_image = resize_image(trimmed_image, TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
        extracted_data = extract_text_from_svg_fields(resized_image)
        cleaned_data = post_process_extracted_data(extracted_data)

        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }

        return JSONResponse(content={"extracted_data": cleaned_data}, headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")

