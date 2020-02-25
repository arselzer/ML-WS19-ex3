#!/usr/bin/env python
import os
import sys

def print_usage():
    print("USAGE:", sys.argv[0],"model_path image_path")
    quit()

if (len(sys.argv) != 3):
    print_usage()

import concurrent.futures
import pandas as pd
import numpy as np
import urllib
import pathlib
import hashlib
#import cv2
import json
import tensorflow as tf
import PIL
import skimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_image(filename):
	img = tf.keras.preprocessing.image.load_img(filename, target_size=(IMG_WIDTH,IMG_HEIGHT))
	img = tf.keras.preprocessing.image.img_to_array(img)
	#img = np.array(img).astype("float32")/255
	#img = skimage.transform.resize(img, (IMG_WIDTH, IMG_HEIGHT, 3))
	img = np.expand_dims(img, axis=0) / 255
	return img

train_data = load_image(sys.argv[2])

labels_file = open("labels.json", "r")
labels = json.loads(labels_file.read())

print(labels)

model = tf.keras.models.load_model(sys.argv[1])

model.summary()

#plot_model(model, to_file='model.png')

#img = load_image(sys.argv[1])

#predictions = model.predict(img, verbose=1)
predictions = model.predict(train_data, verbose=1)
prediction = predictions.argmax(axis=-1)

print(predictions)
print(prediction)
map_labels = np.vectorize(lambda i: labels[str(i)])
print(map_labels(prediction))
#print(labels[str(prediction)])
