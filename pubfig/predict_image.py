#!/usr/bin/env python

import concurrent.futures
import pandas as pd
import numpy as np
import urllib
import pathlib
import hashlib
import os
import sys
import cv2
import json
import tensorflow as tf
import PIL
import skimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 150
IMG_WIDTH = 150


# First recreate the class labels from the directories

#image_generator = ImageDataGenerator(rescale=1./255)
#train_data = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
#	directory=sys.argv[1],
#	target_size=(IMG_HEIGHT, IMG_WIDTH),
#	class_mode=None)

def load_image(filename):
	img = tf.keras.preprocessing.image.load_img(filename, target_size=(IMG_WIDTH,IMG_HEIGHT))
	img = tf.keras.preprocessing.image.img_to_array(img)
	#img = np.array(img).astype("float32")/255
	#img = skimage.transform.resize(img, (IMG_WIDTH, IMG_HEIGHT, 3))
	img = np.expand_dims(img, axis=0) / 255
	return img

from glob import glob
class_names = glob("persons-cropped/*")
class_names = sorted(class_names)
#labels = dict(zip(class_names, range(len(class_names))))

labels_file = open("labels.json", "r")
labels = json.loads(labels_file.read())

print(labels)

model = tf.keras.models.load_model("model.h5")

model.summary()

img = load_image(sys.argv[1])

predictions = model.predict(img, verbose=1)
prediction = predictions.argmax(axis=-1)

print(predictions)
print(prediction)
map_labels = np.vectorize(lambda i: labels[str(i)])
print(map_labels(prediction))
#print(labels[str(prediction)])
