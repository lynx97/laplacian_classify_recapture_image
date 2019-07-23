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
from keras import preprocessing
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
    ddepth = cv2.CV_32F
    kernel_size = 3
    #src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src, ddepth, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    #xxx = abs_dst.shape
    #abs_dst = abs_dst.reshape(xxx[0], xxx[1], 1)
    return abs_dst

TRAINING_DIR = './data'
VALIDATE_DIR = './data'
INPUT_SIZE = (224,224)
BATCH_SIZE = 16
datagen = preprocessing.image.ImageDataGenerator(
    preprocessing_function=laplacian_filter
)

train_generator = datagen.flow_from_directory(
    directory=TRAINING_DIR,
    target_size=INPUT_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

# x, y = train_generator.next()
# print(x[1].shape)
# cv2.imshow("img", x[2])
# # plt.show()
# cv2.waitKey(0)
# print(y)
# cv2.destroyAllWindows()
valid_generator = datagen.flow_from_directory(
    directory=VALIDATE_DIR,
    target_size=INPUT_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
from keras.applications.mobilenet_v2 import MobileNetV2
def InitModel():
    image_size = 224
    #Load the MobileNetV2 model
    mobileNetv2_conv = MobileNetV2(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))

    for layer in mobileNetv2_conv.layers[:-4]:
        layer.trainable = True
    # Create the model
    model = models.Sequential()
    
    # Add the vgg convolutional base model
    model.add(mobileNetv2_conv)
    
    # Add new layers
    model.add(layers.Flatten())
    #model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(2, activation='softmax'))
    
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc', matthews_correlation])
    return model

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = K.callbacks.TensorBoard(log_dir=logdir)

# checkpoint
filepath="checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
Recapture_Model = InitModel()
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = Recapture_Model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=600,
                    callbacks=[tensorboard_callback, checkpoint]
)