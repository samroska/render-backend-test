from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Image Prediction API",
    description="A FastAPI backend that processes PNG images through a machine learning model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://bespoke-medovik-0b9d2c.netlify.app"],
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

        img = Image.open(io.BytesIO(file_content))
        img.verify()
        img = Image.open(io.BytesIO(file_content))

        return JSONResponse(status_code=200, content={"message": "File received successfully",
                                                       "data": {"filename": file.filename,
                                                                 "file_length": len(file_content),
                                                                 "image_format": img.format}})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )