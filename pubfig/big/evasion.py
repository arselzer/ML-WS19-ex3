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
from keras.utils import plot_model

from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy
from cleverhans.utils import AccuracyReport
from cleverhans.utils_tf import model_eval

IMG_HEIGHT = 128
IMG_WIDTH = 128

#image_generator = ImageDataGenerator(rescale=1./255)
#train_data = image_generator.flow_from_directory(batch_size=32,
#	directory=sys.argv[1],
#	target_size=(IMG_HEIGHT, IMG_WIDTH),
#	class_mode=None)

def load_image(filename):
	img = tf.keras.preprocessing.image.load_img(filename, target_size=(IMG_WIDTH,IMG_HEIGHT))
	img = tf.keras.preprocessing.image.img_to_array(img)
	img = np.expand_dims(img, axis=0) / 255
	return img

labels_file = open("labels.json", "r")
labels = json.loads(labels_file.read())

x = load_image(sys.argv[1])

sess=tf.session()

model = tf.keras.models.load_model("model.h5")

model.summary()

#plot_model(model, to_file='model.png')

#img = load_image(sys.argv[1])

fgsm = FastGradientMethod(model)
fgsm_params = {'eps': 0.3,
                 'clip_min': 0.,
                 'clip_max': 1.}
adv_x = fgsm.generate(x, **fgsm_params)
  # Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)
preds_adv = model(adv_x)



#predictions = model.predict(img, verbose=1)
#predictions = model.predict(train_data, verbose=1)
#prediction = predictions.argmax(axis=-1)

print(predictions)
print(prediction)
map_labels = np.vectorize(lambda i: labels[str(i)])
print(map_labels(prediction))
#print(labels[str(prediction)])
