#!/usr/bin/env python
# coding: utf-8

import os
import sys

def print_usage():
    print("USAGE:", sys.argv[0], "train_data_path validation_data_path [model_path]")
    quit()

if len(sys.argv) > 4:
    print_usage()
elif len(sys.argv) == 4:
    TRAIN_DIR = sys.argv[1]
    TEST_DIR = sys.argv[2]
    MODEL_PATH = sys.argv[3]
    print("Training with custom train, test and model paths")
elif len(sys.argv) == 3:
    TRAIN_DIR = sys.argv[1]
    TEST_DIR = sys.argv[2]
    print("Training from scratch with custom training and validation data")
else:
    TRAIN_DIR = "./persons-cropped/"
    TEST_DIR = "./persons-cropped-test/"
    print("Training from scratch from default train/test data dirs")

# # Versuch 4
# ## Mit VGGFace

# In[2]:


import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Flatten, Dropout, Input
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import ResNet50


# In[3]:


from keras.preprocessing.image import ImageDataGenerator
from keras_vggface.utils import preprocess_input

BATCH_SIZE = 10
TEST_BATCH_SIZE = 10
HEIGHT = 224
WIDTH = 224

train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=45,
      horizontal_flip=False,
      vertical_flip=False
    )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE,
                                                   class_mode='categorical',
                                                   color_mode='rgb')

validation_generator = train_datagen.flow_from_directory(TEST_DIR,
                                                        target_size=(HEIGHT,WIDTH),
                                                        batch_size=TEST_BATCH_SIZE,
                                                        class_mode='categorical',
                                                        color_mode='rgb')


# #### Model:

# In[4]:

if not 'MODEL_PATH' in vars():
    from keras_vggface.vggface import VGGFace
    x=Dropout(0.2, input_shape=(224,224,3))
    base_model = VGGFace(include_top=False, input_shape=(224, 224, 3), weights='vggface')
    #base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(HEIGHT,WIDTH,3)) #imports the mobilenet model and discards the last 1000 neuron layer.
    #base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    for layer in base_model.layers:
        layer.trainable=False

    x=base_model.output
    #x=GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    #x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    #x=Dense(1024,activation='relu')(x) #dense layer 2
    #x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(140,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)
    #specify the inputs
    #specify the outputs
    #now a model has been created based on our architecture
    model.compile(optimizer=Adam(lr= 1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

else:
    model=tf.keras.models.load_model(sys.argv[3])

model.summary()


# In[5]:


from keras.callbacks import ModelCheckpoint
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')


filepath="./checkpoints/" + "VGGFace" + "_model_weights_verbose.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["val_acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

step_size_train=train_generator.n//train_generator.batch_size
step_size_validation=validation_generator.n//validation_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=3,
                    validation_data=validation_generator,
                   validation_steps=step_size_validation,
                   callbacks=callbacks_list,
                   verbose=1)


# In[1]:


def plot_training(history):
    print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    ax0 = axes[0]
    ax1 = axes[1]

    ax0.plot(epochs, acc, 'b')
    ax0.plot(epochs, val_acc, 'g')
    ax0.legend(['Training set','Test set'])
    ax0.title.set_text('Training and validation accuracy')
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Accuracy %')
    
    ax1.plot(epochs, loss, 'b')
    ax1.plot(epochs, val_loss, 'g')
    ax1.legend(['Training set','Test set'])
    ax1.title.set_text('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    plt.savefig('acc_vs_epochs_Adam_e4.png')
    plt.show()
    
#plot_training(model.history)

