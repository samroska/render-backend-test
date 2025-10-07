import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras import Model
from PIL import Image
 
loaded_model = tf.keras.models.load_model('PAD-UFES-20.keras')
 
def inference_function(image):
  image_rgb = Image.open(image).convert('RGB')
  validation_image = tf.image.resize(img_to_array(image_rgb),[64,64])
  validation_array = img_to_array(validation_image).reshape(1,64,64,3)
  validation_array = validation_array/255.0
  prediction = loaded_model.predict(validation_array)
  print('\nProbabilities:')
  print('ACK: ' + str(round(prediction[0][0],3)))
  print('BCC: ' + str(round(prediction[0][1],3)))
  print('MEL: ' + str(round(prediction[0][2],3)))
  print('NEV: ' + str(round(prediction[0][3],3)))
  print('SCC: ' + str(round(prediction[0][4],3)))
  print('SEK: ' + str(round(prediction[0][5],3)))
 