# -*- coding: utf-8 -*-
"""RandomForest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YhsupsB6IbqQAC_PXZ_NYLs847MTxW38
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
# %matplotlib notebook
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist


# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Making a copy before flattening for the next code-segment which displays images
x_train_drawing = x_train

image_size = 784 # 28 x 28
x_train = x_train.reshape(x_train.shape[0], image_size) 
x_test = x_test.reshape(x_test.shape[0], image_size)

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train / 255.0 - 0.5
x_test = x_test / 255.0 - 0.5



rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10, random_state=0)
rfc.fit(x_train, y_train)
res1 = rfc.score(x_test, y_test)
res2 = rfc.score(x_train, y_train)
print('Test Accuracy', res1)
print('Train Accuracy', res2)