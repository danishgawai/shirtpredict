from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
# Keras
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

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

f = Path("model_structurevgg16.json")
model_structure = f.read_text()
model = tf.keras.models.model_from_json(model_structure)
model.load_weights("model_weightsvgg16.h5")
#model._make_predict_function()
feature_extraction_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

def img2classify(img_path):
  try: 
    img = Image.open(img_path).convert('RGB')#.convert("L") # read the image using OpenCV
  except IOError: 
    pass
# Convert the image to a numpy array
  rsimg = img.resize((75,75))
  image_array = tf.keras.preprocessing.image.img_to_array(rsimg)

# Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
  images = np.expand_dims(image_array, axis=0)
  names = [ 'full_sleeves', 'half_sleeves','sleeveless', 'three_fourth']
# Normalize the data
  images = tf.keras.applications.vgg16.preprocess_input(images)
  features = feature_extraction_model.predict(images)
  results = model.predict(features)

# Since we are only testing one image with possible class, we only need to check the first result's first element
  results
  single_result = results.argmax()

  #print("Likelihood that this image contains an" )
#for single_result in results:
  #return print(names[single_result])
  return names[single_result]

#names = [ 'full_sleeves', 'half_sleeves','sleeveless', 'three_fourth']

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        names = [ 'full_sleeves', 'half_sleeves','sleeveless', 'three_fourth']
        # Make prediction
        preds = img2classify(file_path)

        #preds = prediction.argmax()  # Simple argmax
        # pred_class = tf.keras.applications.vgg16.decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        result = str(preds)
        return result
    return None

if __name__ == '__main__':
    app.run(port=5000,debug=True)


