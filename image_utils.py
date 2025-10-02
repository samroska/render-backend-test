from PIL import Image
import io
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
     
    
    @staticmethod
    def validate_image_format(file_content: bytes) -> bool:
       
        try:
            image = Image.open(io.BytesIO(file_content))
            return image.format.upper() in ['PNG', 'JPEG']
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False
    
    @staticmethod
    def load_image_from_bytes(file_content: bytes) -> Image.Image:
         
        try:
            image = Image.open(io.BytesIO(file_content))
            # Convert to RGB if needed (removes alpha channel)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Failed to load image: {e}")
    
    @staticmethod
    def image_to_numpy(image: Image.Image) -> np.ndarray:

        return np.array(image)
    
    @staticmethod
    def numpy_to_image(array: np.ndarray) -> Image.Image:
         
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
         
        if len(array.shape) == 2:   
            return Image.fromarray(array, mode='L')
        elif len(array.shape) == 3 and array.shape[2] == 3:  
            return Image.fromarray(array, mode='RGB')
        elif len(array.shape) == 3 and array.shape[2] == 4:   
            return Image.fromarray(array, mode='RGBA')
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
  
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    
    @staticmethod
    def resize_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
  
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def validate_image_size(image: Image.Image, max_size: Tuple[int, int] = (2048, 2048)) -> bool:
 
        width, height = image.size
        max_width, max_height = max_size
        return width <= max_width and height <= max_height