from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = r'C:/Users/Yagna/Desktop/Final Year Project/Brain-Tumor-Detection-master/mnist-model.h5'

# Load your trained model
        # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications

model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    img = cv2.imread(img_path)
# Using cv2.imread() method
# Using 0 to read image in grayscale mode
    img2 = cv2.resize(img, dsize=(240,240), interpolation=cv2.INTER_CUBIC)
    img2 = img2.reshape((1, 240, 240, 3))
    preds = model.predict(img2, batch_size=1)
    return preds


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

        # Make prediction
        preds = model_predict(file_path, model)
        ans = (preds[0][0])
        if ans >= 1 : 
           output = "Tumor Detected"
        else:
            output = "Tumor Not Detected"
        return output
    return 'Error'



if __name__ == '__main__':
    app.run(debug=True)
