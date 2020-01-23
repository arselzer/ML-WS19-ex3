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

train_data = image_generator.flow_from_directory(batch_size=64,
	directory="persons-cropped",
	shuffle=True,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	class_mode="categorical",
	subset="training")

validation_data = image_generator.flow_from_directory(batch_size=64,
	directory="persons-cropped",
	shuffle=True,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	class_mode="categorical",
	subset="validation")
