# Load neccessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications import vgg16
from keras.preprocessing import image
import cv2
import os                   
import joblib
from pathlib import Path
from PIL import Image, ImageFile

names = [ 'full_sleeves', 'half_sleeves','sleeveless', 'three_fourth']

f = Path("model_structurevgg16.json")
model_structure = f.read_text()
model = keras.models.model_from_json(model_structure)

model.load_weights("model_weightsvgg16.h5")

feature_extraction_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

def img2classify(img_path):
  try: 
    img = Image.open(img_path).convert('RGB')#.convert("L") # read the image using OpenCV
  except IOError: 
    pass
# Convert the image to a numpy array
  rsimg = img.resize((75,75))
  image_array = image.img_to_array(rsimg)

# Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
  images = np.expand_dims(image_array, axis=0)

# Normalize the data
  images = vgg16.preprocess_input(images)
  features = feature_extraction_model.predict(images)
  results = model.predict(features)

# Since we are only testing one image with possible class, we only need to check the first result's first element
  results
  single_result = results.argmax()

  print("Likelihood that this image contains an" )
#for single_result in results:
  return print(names[single_result])

img2classify('uploads/half1.jpg')
