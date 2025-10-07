import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import logging
from typing import Dict, Union
import os
import zipfile
import tempfile
import shutil
import glob

logger = logging.getLogger(__name__)

class SkinLesionClassifier:
    """
    A class for skin lesion classification using a pre-trained Keras model.
    
    This class loads a trained model and provides methods for preprocessing
    images and making predictions on skin lesion images.
    """
 
    def __init__(self, model_path: str = 'PAD-UFES-20.keras'):
        """
        Initialize the classifier with a pre-trained model.
        
        Args:
            model_path (str): Path to the trained Keras model file or zip file containing the model
        """
        self.original_model_path = model_path
        self.model_path = model_path
        self.model = None
        self.class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        self.input_size = (64, 64)
        self.temp_dir = None
        
        self._load_model()
    
    def _reassemble_split_files(self, base_path: str) -> str:
        """Reassemble split files if they exist."""
        # Check if split files exist
        split_pattern = f"{base_path}.part*"
        split_files = sorted(glob.glob(split_pattern))
        
        if not split_files:
            return base_path  # No split files found, return original path
        
        logger.info(f"Found {len(split_files)} split files: {split_files}")
        
        # Create temporary file for reassembled content
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zip')
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                for split_file in split_files:
                    logger.info(f"Reading split file: {split_file}")
                    with open(split_file, 'rb') as sf:
                        temp_file.write(sf.read())
            
            logger.info(f"Successfully reassembled {len(split_files)} files into {temp_path}")
            return temp_path
            
        except Exception as e:
            # Clean up temp file if something went wrong
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e
    
    def _extract_model_if_zipped(self):
        """Extract model from zip file if needed, handling split files."""
        original_path = self.original_model_path
        
        # Check if we need to reassemble split files
        if not os.path.exists(original_path):
            # Try to find and reassemble split files
            reassembled_path = self._reassemble_split_files(original_path)
            if reassembled_path != original_path:
                self.model_path = reassembled_path
                original_path = reassembled_path
        
        # Check if the file is a zip file
        if original_path.endswith('.zip') or self.original_model_path.endswith('.zip'):
            try:
                logger.info(f"Extracting model from zip: {original_path}")
                
                # Create a temporary directory
                if not self.temp_dir:
                    self.temp_dir = tempfile.mkdtemp()
                
                # Extract the zip file
                with zipfile.ZipFile(original_path, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_dir)
                
                # Find the model file in the extracted content
                for root, dirs, files in os.walk(self.temp_dir):
                    for file in files:
                        if file.endswith('.keras') or file.endswith('.h5'):
                            self.model_path = os.path.join(root, file)
                            logger.info(f"Found model file: {self.model_path}")
                            return
                
                raise FileNotFoundError("No .keras or .h5 model file found in the zip archive")
                
            except Exception as e:
                logger.error(f"Error extracting model from zip: {e}")
                raise
    
    def _load_model(self):
        """Load the pre-trained model from disk."""
        try:
            # Check if we need to extract from zip
            self._extract_model_if_zipped()
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Clean up temp directory if it was created
            self._cleanup_temp_files()
            raise
    
    def _cleanup_temp_files(self):
        """Clean up temporary files if they exist."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")
    
    def __del__(self):
        """Destructor to clean up temporary files."""
        self._cleanup_temp_files()
    
    def preprocess_image(self, image: Union[Image.Image, str]) -> np.ndarray:
 
        try:
            # Handle both PIL Image objects and file paths
            if isinstance(image, str):
                image_rgb = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image_rgb = image.convert('RGB')
            else:
                raise ValueError("Image must be a PIL Image object or file path")
            
            # Convert to array and resize
            image_array = img_to_array(image_rgb)
            resized_image = tf.image.resize(image_array, self.input_size)
            
            # Reshape and normalize
            processed_array = img_to_array(resized_image).reshape(1, 64, 64, 3)
            processed_array = processed_array / 255.0
            
            return processed_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image: Union[Image.Image, str]) -> Dict[str, float]:
 
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call _load_model() first.")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Create results dictionary
            results = {}
            for i, class_name in enumerate(self.class_names):
                results[class_name] = float(round(prediction[0][i], 3))
            
            logger.info(f"Prediction completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def get_top_prediction(self, image: Union[Image.Image, str]) -> tuple:
  
        predictions = self.predict(image)
        top_class = max(predictions, key=predictions.get)
        confidence = predictions[top_class]
        
        return top_class, confidence
    
    def print_predictions(self, image: Union[Image.Image, str]):
 
        predictions = self.predict(image)
        
        print('\nProbabilities:')
        for class_name, probability in predictions.items():
            print(f'{class_name}: {probability}')
    
    def get_prediction_summary(self, image: Union[Image.Image, str]) -> Dict:
 
        predictions = self.predict(image)
        top_class, confidence = self.get_top_prediction(image)
        
        return {
            'top_prediction': {
                'class': top_class,
                'confidence': confidence
            },
            'all_predictions': predictions,
            'model_info': {
                'classes': self.class_names,
                'input_size': self.input_size
            }
        }