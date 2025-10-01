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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Image Prediction API",
    description="A FastAPI backend that processes PNG images through a machine learning model",
    version="1.0.0"
)

# Configure CORS for React.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "ML Image Prediction API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "ml-image-api"}

@app.get("/model/info")
async def get_model_info():
    """Get information about the ML model"""
    return ml_model.get_model_info()

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Test endpoint to debug file uploads"""
    try:
        logger.info(f"Test upload - filename: {file.filename}, content_type: {file.content_type}")
        content = await file.read()
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Test upload error: {e}")
        return {"error": str(e), "status": "failed"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(..., description="PNG image file to process")):
    """
    Process an uploaded PNG image through the ML model and return the result as PNG.
    
    Args:
        file: Uploaded PNG image file
        
    Returns:
        StreamingResponse: Processed image as PNG
    """
    try:
        # Log the incoming request details
        logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
        
        # Check if file was actually uploaded
        if not file or not file.filename:
            logger.error("No file uploaded")
            raise HTTPException(
                status_code=422,
                detail="No file uploaded. Please provide a file."
            )
        
        # Read file content
        file_content = await file.read()
        logger.info(f"File content read successfully, size: {len(file_content)} bytes")
        
        # Check if file content is empty
        if not file_content:
            logger.error("Uploaded file is empty")
            raise HTTPException(
                status_code=422,
                detail="Uploaded file is empty"
            )
        
        # Try to validate the image by opening it with PIL first
        try:
            # Use PIL to validate the image
            image_test = Image.open(io.BytesIO(file_content))
            image_test.verify()  # This will raise an exception if it's not a valid image
            logger.info(f"Image validated successfully: {image_test.format}")
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Validate PNG format using our utility (optional, since we already validated above)
        if not ImageProcessor.validate_image_format(file_content):
            logger.error("File is not a valid PNG image")
            raise HTTPException(
                status_code=422,
                detail="File must be a valid PNG image"
            )
        
        # Load and validate image
        image = ImageProcessor.load_image_from_bytes(file_content)
        
        # Check image size limits
        if not ImageProcessor.validate_image_size(image):
            raise HTTPException(
                status_code=400,
                detail="Image too large. Maximum size is 2048x2048 pixels"
            )
        
        logger.info(f"Processing image: {image.size} pixels, mode: {image.mode}")
        
        # Convert image to numpy array for ML model
        image_array = ImageProcessor.image_to_numpy(image)
        
        # Run ML model prediction
        processed_array = ml_model.predict(image_array)
        
        # Convert result back to PIL Image
        processed_image = ImageProcessor.numpy_to_image(processed_array)
        
        # Convert to bytes for response
        output_bytes = ImageProcessor.image_to_bytes(processed_image, format='PNG')
        
        logger.info(f"Successfully processed image. Output size: {processed_image.size}")
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=processed_image.png"
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )