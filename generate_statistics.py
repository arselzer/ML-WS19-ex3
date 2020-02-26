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
import matplotlib.pyplot as plt
import keras
import scipy
import sys

from keras.preprocessing.image import ImageDataGenerator

import foolbox

def print_usage():
    print("USAGE:", sys.argv[0],"model_path method")
    quit()

if (len(sys.argv) != 3):
    print_usage()

IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_image(filename):
	img = tf.keras.preprocessing.image.load_img(filename, target_size=(IMG_WIDTH,IMG_HEIGHT))
	img = tf.keras.preprocessing.image.img_to_array(img)
	img = np.expand_dims(img, axis=0) / 255
	return img

labels_file = open("labels.json", "r")
labels = json.loads(labels_file.read())

inverted_labels = {v: k for k, v in labels.items()}

model = tf.keras.models.load_model(sys.argv[1])

fmodel = foolbox.models.KerasModel(model=model, bounds=(0.0,1.0), predicts="logits")

model.summary()

img = load_image("persons-cropped/Donald Trump/122.jpg")


# In[42]:


train_classes =os.listdir("persons-cropped")
test_classes = os.listdir("persons-cropped-test")

attack = foolbox.attacks.SaliencyMapAttack(fmodel)

def create_adv(img, label):
    if sys.argv[2] == "gaussian":
        adv = scipy.ndimage.gaussian_filter(img, sigma=5)
        return adv
    if sys.argv[2] == "saliency":
        adv = attack(img, np.array([label]))
        return adv

print("train set:")
correct = 0
total = 0
for person in train_classes:
    for file in os.listdir("persons-cropped/" + person):
        img = load_image("persons-cropped/" + person + "/" + file)
        adv = create_adv(img, int(inverted_labels[person]))
        preds = model.predict(adv)
        pred = tf.argmax(preds, axis=1)[0].numpy()
        print(person == labels[str(pred)])
        
        if person == labels[str(pred)]:
            correct += 1
        total += 1
print ("correct: " + str(correct) + "(" + str(correct / total) + ")" + " wrong: " + str(total - correct) + "(" + str((total - correct) / total) + ")")
    
print("test set:")
correct = 0
total = 0
for person in test_classes:
    for file in os.listdir("persons-cropped-test/" + person):
        img = load_image("persons-cropped-test/" + person + "/" + file)
        adv = create_adv(img, int(inverted_labels[person]))
        preds = model.predict(adv)
        pred = tf.argmax(preds, axis=1)[0].numpy()
        
        if person == labels[str(pred)]:
            correct += 1
        total += 1
print ("correct: " + str(correct) + "(" + str(correct / total) + ")" + " wrong: " + str(total - correct) + "(" + str((total - correct) / total) + ")")


# In[ ]:




