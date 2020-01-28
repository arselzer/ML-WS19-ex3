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
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

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
	#img = np.array(img).astype("float32")/255
	#img = skimage.transform.resize(img, (IMG_WIDTH, IMG_HEIGHT, 3))
	img = np.expand_dims(img, axis=0) / 255
	return img

train_data = load_image(sys.argv[1])

labels_file = open("labels.json", "r")
labels = json.loads(labels_file.read())

#print(labels)

pretrained_model = tf.keras.models.load_model("model.h5")
pretrained_model.trainable = False

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

image_probs = pretrained_model.predict(train_data)

## HIER DIE PREDICTION CONFIDENCE

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
	with tf.GradientTape() as tape:
		tape.watch(input_image)
		prediction = pretrained_model(input_image)
		loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
	gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
	signed_grad = tf.sign(gradient)
	return signed_grad

perturbations = create_adversarial_pattern(train_data, 10)

epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
	adv_x = image + eps*perturbations
	adv_x = tf.clip_by_value(adv_x, 0, 1)
	display_images(adv_x, descriptions[i])
	adv_x.save("img_" + descriptions[i] + ".jpg")



#model.summary()

#plot_model(model, to_file='model.png')

#img = load_image(sys.argv[1])

#predictions = model.predict(img, verbose=1)
#predictions = model.predict(train_data, verbose=1)
#prediction = predictions.argmax(axis=-1)

#print(predictions)
#print(prediction)
#map_labels = np.vectorize(lambda i: labels[str(i)])
#print(map_labels(prediction))
#print(labels[str(prediction)])
