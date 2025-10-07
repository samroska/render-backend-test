import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import logging
from typing import Dict, Union
import os

logger = logging.getLogger(__name__)

class SkinLesionClassifier:
 
    def __init__(self, model_path: str = 'PAD-UFES-20.keras'):
        
        self.model_path = model_path
        self.model = None
        self.class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        self.input_size = (64, 64)
        
        self._load_model()
    
    def _load_model(self):
         
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
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