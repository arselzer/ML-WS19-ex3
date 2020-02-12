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

#people = pd.read_csv("dev_people.txt")

labels_file = open("labels.json", "r")
labels = json.loads(labels_file.read())

#print(labels)

model = tf.keras.models.load_model("model.h5")

image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.1, rotation_range=45, zoom_range=0.2)

IMG_HEIGHT = 128
IMG_WIDTH = 128
LEARNING_RATE = 0.0003
BATCH_SIZE = 32
NUM_TRAIN = 100
STEPS_PER_EPOCH = round(NUM_TRAIN) // BATCH_SIZE
VAL_STEPS = 20
NUM_EPOCHS = 45

train_data = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
	directory="persons-adversarial-odds",
	shuffle=True,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	class_mode="categorical",
	subset="training")

labels = {**labels, **train_data.class_indices}
labels = {v: k for k, v in labels.items()}

with open("labels_advproof.json", "w") as labels_file:
	labels_file.write(json.dumps(labels))

validation_data = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
	directory="persons-adversarial-odds",
	shuffle=True,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	class_mode="categorical",
	subset="validation")

model.compile(
	optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
	loss="categorical_crossentropy",
	metrics=["accuracy"]
)

model.summary()

history = model.fit(
	train_data,
	epochs=NUM_EPOCHS,
	steps_per_epoch=None,
	validation_data=validation_data,
	validation_steps=None,
	use_multiprocessing=False,
	workers=8,
	verbose=2
)

print(history)

model.save("model_advproof.h5")
