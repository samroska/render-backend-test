from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from typing import Optional
import logging

from image_utils import ImageProcessor
from ml_model import ml_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Image Prediction API",
    description="A FastAPI backend that processes PNG images through a machine learning model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "https://bespoke-medovik-0b9d2c.netlify.app"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "ML Image Prediction API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-image-api"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    try:
 
        logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
 
        if not file or not file.filename:
            logger.error("No file uploaded")
            raise HTTPException(
                status_code=422,
                detail="No file uploaded. Please provide a file."
            )

        file_content = await file.read()
        logger.info(f"File content read successfully, size: {len(file_content)} bytes")

        if not file_content:
            logger.error("Uploaded file is empty")
            raise HTTPException(
                status_code=422,
                detail="Uploaded file is empty"
            )
 
        try:
            image_test = Image.open(io.BytesIO(file_content))
            image_test.verify()
            logger.info(f"Image validated successfully: {image_test.format}")
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid image file: {str(e)}"
            )
        
        if not ImageProcessor.validate_image_format(file_content):
            logger.error("File is not a valid PNG or JPEG image")
            raise HTTPException(
                status_code=422,
                detail="File must be a valid PNG or JPEG image"
            )
 
        image = ImageProcessor.load_image_from_bytes(file_content)
 
        if not ImageProcessor.validate_image_size(image):
            raise HTTPException(
                status_code=400,
                detail="Image too large. Maximum size is 2048x2048 pixels"
            )
        
        logger.info(f"Processing image: {image.size} pixels, mode: {image.mode}")
        
        image_array = ImageProcessor.image_to_numpy(image)
        
        processed_array = ml_model.predict(image_array)
        
        processed_image = ImageProcessor.numpy_to_image(processed_array)
        
        output_bytes = ImageProcessor.image_to_bytes(processed_image, format='PNG')
        
        logger.info(f"Successfully processed image. Output size: {processed_image.size}")
        
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=processed_image.png"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )