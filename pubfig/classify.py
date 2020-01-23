#!/usr/bin/env python

import concurrent.futures
import pandas as pd
import urllib
import pathlib
import hashlib
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

people = pd.read_csv("dev_people.txt")

image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

IMG_HEIGHT = 150
IMG_WIDTH = 150
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 30
NUM_TRAIN = 100
STEPS_PER_EPOCH = round(NUM_TRAIN) // BATCH_SIZE
VAL_STEPS = 20

train_data = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
	directory="persons-cropped",
	shuffle=True,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	class_mode="categorical",
	subset="training")

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
prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")

model = tf.keras.Sequential([
	base_model,
	maxpool_layer,
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
	steps_per_epoch=STEPS_PER_EPOCH,
	validation_data=validation_data,
	validation_steps=VAL_STEPS)

