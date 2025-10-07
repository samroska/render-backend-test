from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from typing import Optional
import logging
import skin_lesion_classifier as Processor
from ml_model import inference_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Image Prediction API",
    description="A FastAPI backend that processes PNG images through a machine learning model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # "http://localhost:5173",
        # "http://127.0.0.1:5173", 
        "https://bespoke-medovik-0b9d2c.netlify.app"
    ],
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
async def predict_image(file: UploadFile = File(..., description="PNG image file to process")):
 
    try:
        logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
        
        # Validate file upload
        if not file or not file.filename:
            logger.error("No file uploaded")
            return JSONResponse(
                status_code=422,
                content={"error": "No file uploaded. Please provide a PNG image file."}
            )
        
        # Read file content
        file_content = await file.read()
        logger.info(f"File content read successfully, size: {len(file_content)} bytes")
        
        # Validate file content is not empty
        if not file_content:
            logger.error("Uploaded file is empty")
            return JSONResponse(
                status_code=422,
                content={"error": "Uploaded file is empty. Please provide a valid PNG image."}
            )
        
        # Validate image format
        try:
            img = Image.open(io.BytesIO(file_content))
            img.verify()  # Verify the image is valid
            
            # Re-open the image after verify (verify() can corrupt the image object)
            img = Image.open(io.BytesIO(file_content))
            
            # Check if the image format is PNG
            if img.format != 'PNG':
                logger.error(f"Invalid image format: {img.format}. Expected PNG.")
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": f"Invalid image format: {img.format}. Only PNG images are accepted.",
                        "received_format": img.format,
                        "expected_format": "PNG"
                    }
                )
            
            logger.info(f"Valid PNG image validated: {img.format}, size: {img.size}")
            
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return JSONResponse(
                status_code=422,
                content={"error": "Invalid image file. Please upload a valid PNG image."}
            )
        
        # Process image through ML model
        try:
            # Use the inference function from ml_model
            predictions = inference_function(img)
            
            # Get the top prediction
            top_class = max(predictions, key=predictions.get)
            confidence = predictions[top_class]
            
            logger.info(f"Prediction completed successfully for {file.filename}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Image processed successfully",
                    "filename": file.filename,
                    "image_info": {
                        "format": img.format,
                        "size": img.size,
                        "mode": img.mode,
                        "file_size_bytes": len(file_content)
                    },
                    "predictions": {
                        "top_prediction": {
                            "class": top_class,
                            "confidence": confidence
                        },
                        "all_probabilities": predictions
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error during ML inference: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing image through ML model: {str(e)}"}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing image: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )