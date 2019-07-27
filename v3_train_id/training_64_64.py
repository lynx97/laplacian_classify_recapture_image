import collections
import io
import math
import os
import random
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
from datetime import datetime
import keras as K
from keras import models
from keras import layers
from keras import optimizers
from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import preprocessing
from keras.models import load_model
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.backend.round(K.backend.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.backend.round(K.backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.backend.sum(y_pos * y_pred_pos)
    tn = K.backend.sum(y_neg * y_pred_neg)

    fp = K.backend.sum(y_neg * y_pred_pos)
    fn = K.backend.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.backend.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.backend.epsilon())

def laplacian_filter(src):
    #src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.float32)
    kernel[0][0] = 0
    kernel[0][1] = -1
    kernel[0][2] = 0
    kernel[1][0] = -1
    kernel[1][1] = 4
    kernel[1][2] = -1
    kernel[2][0] = 0
    kernel[2][1] = -1
    kernel[2][2] = 0
    dst = cv2.filter2D(src,-1,kernel)
    dst = np.expand_dims(dst, axis=-1)
    return dst

# TRAINING_DIR = '/home/dev/liemhd/haihh/recapture/dataset/64x64data/training'
# VALIDATE_DIR = '/home/dev/liemhd/haihh/recapture/dataset/64x64data/validation'
TRAINING_DIR = './64x64_id/training'
VALIDATE_DIR = './64x64_id/validation'
ttt = "/home/liem/hai/recapture_classification/test"
INPUT_SIZE = (64,64)
BATCH_SIZE = 32
datagen = preprocessing.image.ImageDataGenerator(
    preprocessing_function=laplacian_filter
)

train_generator = datagen.flow_from_directory(
    directory=TRAINING_DIR,
    target_size=INPUT_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)
valid_generator = datagen.flow_from_directory(
    directory=VALIDATE_DIR,
    target_size=INPUT_SIZE,
    color_mode="grayscale",
    batch_size=256,
    class_mode="categorical",
    shuffle=True
)
# def InitModel():
#     rcp_model = K.models.Sequential()
#     rcp_model.add(Conv2D(64, (3, 3), strides = (1, 1), name = 'conv0', padding="same", input_shape=(64,64,1)))
#     rcp_model.add(BatchNormalization(axis = -1))
#     rcp_model.add(Activation('relu'))
#     rcp_model.add(AveragePooling2D((5, 5), padding="same", strides=(2,2)))

#     #CONV2
#     rcp_model.add(Conv2D(128, (3, 3), strides = (1, 1), name = 'conv1', padding="same"))
#     rcp_model.add(BatchNormalization(axis = -1))
#     rcp_model.add(Activation('relu'))
#     rcp_model.add(AveragePooling2D((5, 5), padding="same", strides=(2,2)))

#     #CONV3
#     rcp_model.add(Conv2D(256, (3, 3), strides = (1, 1), name = 'conv2', padding="same"))
#     rcp_model.add(BatchNormalization(axis = -1))
#     rcp_model.add(Activation('relu'))
#     rcp_model.add(AveragePooling2D((5, 5), padding="same", strides=(2,2)))
#     #CONV5
#     rcp_model.add(Conv2D(256, (3, 3), strides = (1, 1), name = 'conv4', padding="same"))
#     rcp_model.add(BatchNormalization(axis = -1))
#     rcp_model.add(Activation('relu'))
#     rcp_model.add(K.layers.GlobalAveragePooling2D())
#     rcp_model.add(K.layers.Dense(units=2, activation='softmax'))
#     rcp_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc',matthews_correlation])
#     rcp_model.summary()
#     return rcp_model

from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"]="3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)  #chay server thi comment
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default

#event
logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = K.callbacks.TensorBoard(log_dir=logdir)
# checkpoint
filepath="checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#load model
custom_objects={'matthews_correlation':matthews_correlation}

Recapture_Model = load_model("./trained_models/weights-improvement-84-0.98.hdf5", custom_objects = custom_objects)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history = Recapture_Model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1000,
                    callbacks=[tensorboard_callback, checkpoint]
)
