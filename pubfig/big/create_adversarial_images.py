from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from keras.preprocessing.image import array_to_img
from PIL import Image
from skimage.io import imsave

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.models.load_model("model.h5")

labels_file = open("labels.json", "r")
labels = json.loads(labels_file.read())

pretrained_model.trainable = False

directory = "./persons-cropped/"
epsilons = [0.01, 0.05]

labels_rev = {v: k for k, v in labels.items()}

def preprocess(image):
	image = tf.cast(image, tf.float32)
	image = image/255
	image = tf.image.resize(image, (128, 128))
	image = image[None, ...]
	return image

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

def predict_image(image):
	predictions = pretrained_model.predict(image)
	prediction = predictions.argmax(axis=-1)
	map_labels = np.vectorize(lambda i: labels[str(i)])
	return str(map_labels(prediction)[0])

if not (os.path.isdir(os.path.join(directory))):
	print("Directory does not exist")

if not (os.path.isdir("./persons-adversarial")):
	os.mkdir("./persons-adversarial")

total = 0
results = []
img_nr = 0
for person in os.listdir(directory):
	for image in os.listdir(directory + "/" + person):
		total += 1
		# Decode and preprocess image
		image_raw = tf.io.read_file(os.path.join(directory, person, image))
		image = tf.image.decode_image(image_raw)
		image = preprocess(image)
		try:
			image_probs = pretrained_model.predict(image)
		except:
			continue
		image_results = []

		# get label of image
		label_index = int(labels_rev[person])
		label = tf.one_hot(label_index, image_probs.shape[-1])
		label = tf.reshape(label, (1, image_probs.shape[-1]))

		if not (os.path.isdir(os.path.join("./persons-adversarial/", person))):
			os.mkdir(os.path.join("./persons-adversarial/", person))

		perturbations = create_adversarial_pattern(image, label)
		for eps in epsilons:
			adv_x = image + eps*perturbations
			adv_x = tf.clip_by_value(adv_x, 0, 1)

			#prediction = predict_image(adv_x)

			path_to_save = os.path.join("./persons-adversarial/", person, str(img_nr) + ".jpg")
			img_nr += 1
			imsave(path_to_save, adv_x[0])

			#prediction = predict_image(adv_x)
			#image_results.append(prediction == person)
			#print("--------------------------------")
			#print("eps: ", eps)
			#print("Prediction: ", prediction)
			#print("Actually: ", person)
			#print("Result: ", (prediction == person))
			#print("--------------------------------")
			#print("Result: " + str(prediction == person) + ", eps=" + str(eps) +  ", Prediction: " + prediction + ", Actually: " + person)
		results.append(image_results)

print("Total: ", total)
print(np.sum(np.array(results), axis=0))
