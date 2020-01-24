#!/usr/bin/env python

import concurrent.futures
import pandas as pd
import urllib
import pathlib
import hashlib
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

people = pd.read_csv("dev_people.txt")

image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2, rotation_range=45, zoom_range=0.2)

IMG_HEIGHT = 150
IMG_WIDTH = 150
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_TRAIN = 100
STEPS_PER_EPOCH = round(NUM_TRAIN) // BATCH_SIZE
VAL_STEPS = 20
NUM_EPOCHS = 3

train_data = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
	directory="persons-cropped",
	shuffle=True,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	class_mode="categorical",
	subset="training")

labels = train_data.class_indices
labels = {v: k for k, v in labels.items()}

with open("labels.json", "w") as labels_file:
	labels_file.write(json.dumps(labels))

validation_data = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
	directory="persons-cropped",
	shuffle=True,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	class_mode="categorical",
	subset="validation")

base_model = tf.keras.applications.MobileNetV2(
	input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
	include_top=False,
	weights="imagenet"
)

base_model.trainable = False

maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(60, activation="sigmoid")
dropout_layer = tf.keras.layers.Dropout(0.2)

model = tf.keras.Sequential([
	base_model,
	maxpool_layer,
#	dropout_layer,
	prediction_layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
	loss="binary_crossentropy",
	metrics=["accuracy"]
)

model.summary()

model.fit(
	train_data,
	epochs=NUM_EPOCHS,
	steps_per_epoch=None,
	validation_data=validation_data,
	validation_steps=None,
	use_multiprocessing=False,
	workers=6,
	verbose=2
)

model.save("model.h5")
