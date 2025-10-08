from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from typing import Optional
import logging
import skin_lesion_classifier as Processor
from skin_lesion_classifier import SkinLesionClassifier

# Add TensorFlow logging control at the top
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Image Prediction API",
    description="A FastAPI backend that processes PNG, JPG, JPEG, and MPO images through a machine learning model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bespoke-medovik-0b9d2c.netlify.app",
    ],  
    allow_credentials=False, 
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def cors_handler(request: Request, call_next):
    origin = request.headers.get("origin")
    
    if request.method == "OPTIONS":
        response = JSONResponse(content={"message": "OK"})
    else:
        response = await call_next(request)
    
    response.headers["Access-Control-Allow-Origin"] = origin or "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    response.headers["Vary"] = "Origin"
    
    return response

@app.get("/")
async def root():
    return {"message": "ML Image Prediction API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-image-api"}

def convert_mpo_to_png(img):
    """Convert MPO image to PNG format"""
    try:
        # For MPO files, we want the first/main image
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            # MPO files can have multiple frames, get the first one
            img.seek(0)  # Go to first frame
            logger.info(f"MPO file detected with {img.n_frames} frames, using first frame")
        
        # Convert to RGB if needed (removes alpha channel issues)
        if img.mode in ['RGBA', 'LA', 'P']:
            # Create white background for transparent images
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ['RGBA', 'LA']:
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to PNG in memory
        png_buffer = io.BytesIO()
        img.save(png_buffer, format='PNG')
        png_buffer.seek(0)
        
        # Return the PNG image
        return Image.open(png_buffer)
        
    except Exception as e:
        logger.error(f"Error converting MPO to PNG: {e}")
        raise

@app.post("/predict")
async def predict_image(file: UploadFile = File(..., description="PNG, JPG, JPEG, or MPO image file to process")):
    try:
        logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
        
        if not file or not file.filename:
            logger.error("No file uploaded")
            return JSONResponse(
                status_code=422,
                content={"error": "No file uploaded. Please provide a PNG, JPG, JPEG, or MPO image file."},
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        file_content = await file.read()
        logger.info(f"File content read successfully, size: {len(file_content)} bytes")
        
        if not file_content:
            logger.error("Uploaded file is empty")
            return JSONResponse(
                status_code=422,
                content={"error": "Uploaded file is empty. Please provide a valid image file."},
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        # Image validation and conversion
        try:
            img = Image.open(io.BytesIO(file_content))
            img.verify()  # Verify the image is valid
            
            # Re-open after verify (verify can close the image)
            img = Image.open(io.BytesIO(file_content))
            original_format = img.format
            logger.info(f"Detected image format: {original_format}")
            
            # Check if format is supported (now including MPO)
            supported_formats = ['PNG', 'JPEG', 'MPO']
            if img.format not in supported_formats:
                logger.error(f"Unsupported image format: {img.format}")
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": f"Unsupported image format: {img.format}. Only PNG, JPG, JPEG, and MPO are supported.",
                        "received_format": img.format,
                        "supported_formats": supported_formats
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            # Convert to PNG if needed
            if img.format in ['JPEG', 'MPO']:
                logger.info(f"Converting {img.format} to PNG for processing")
                
                if img.format == 'MPO':
                    # Handle MPO specifically
                    img = convert_mpo_to_png(img)
                else:
                    # Handle regular JPEG
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    png_buffer = io.BytesIO()
                    img.save(png_buffer, format='PNG')
                    png_buffer.seek(0)
                    img = Image.open(png_buffer)
                
                logger.info(f"Successfully converted {original_format} to PNG")
            
            logger.info(f"Valid image processed: format={original_format}, final_format=PNG, size={img.size}, mode={img.mode}")
            
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return JSONResponse(
                status_code=422,
                content={"error": "Invalid image file. Please upload a valid PNG, JPG, JPEG, or MPO image."},
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        # ML Model Processing
        try:
            logger.info("Starting ML model inference...")
            predictions = SkinLesionClassifier.predict(img)
            
            if not isinstance(predictions, dict):
                logger.error(f"Invalid predictions format: {type(predictions)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "ML model returned invalid prediction format"},
                    headers={
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            try:
                top_class = max(predictions, key=predictions.get)
                confidence = predictions[top_class]
                logger.info(f"Top prediction: {top_class} with confidence: {confidence}")
            except Exception as e:
                logger.error(f"Error processing predictions: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Error processing model predictions"},
                    headers={
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            logger.info(f"Prediction completed successfully for {file.filename}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Image processed successfully",
                    "filename": file.filename,
                    "image_info": {
                        "original_format": original_format,
                        "processed_format": "PNG",
                        "size": img.size,
                        "mode": img.mode,
                        "file_size_bytes": len(file_content),
                        "converted": original_format != "PNG",
                        "source_type": "iPhone MPO" if original_format == "MPO" else original_format
                    },
                    "predictions": {
                        "top_prediction": {
                            "class": top_class,
                            "confidence": confidence
                        },
                        "all_probabilities": predictions
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization"
                }
            )
            
        except Exception as e:
            logger.error(f"Error during ML inference: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing image through ML model: {str(e)}"},
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing image: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"},
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )