import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt
import tensorflow_datasets as tfds
import os

labels = np.array(['go', 'yes', 'down', 'right', 'up', 'left', 'no', 'stop'])

trainingPath = '/home/jonatank/Documents/inter/spec_data/training'
testingPath = '/home/jonatank/Documents/inter/spec_data/testing'

trainingDirList = os.listdir(trainingPath)
testingDirList = os.listdir(testingPath)


# load training
trainingSpec = []
trainingLabels = []

for dirName in trainingDirList:
    tempFileList = os.listdir(f'{trainingPath}/{dirName}')
    for fname in tempFileList:
        trainingSpec.append(np.load(f'{trainingPath}/{dirName}/{fname}').astype(np.single))
        trainingLabels.append(np.where(labels == dirName))

trainingSpec = np.array(trainingSpec)
trainingLabels = np.array(trainingLabels).flatten()


# load testing
testingSpec = []
testingLabels = []

for dirName in testingDirList:
    tempFileList = os.listdir(f'{testingPath}/{dirName}')
    for fname in tempFileList:
        testingSpec.append(np.load(f'{testingPath}/{dirName}/{fname}').astype(np.single))
        testingLabels.append(np.where(labels == dirName))

testingSpec = np.array(testingSpec)
testingLabels = np.array(testingLabels).flatten()


print(f'training set is shape: {np.shape(trainingSpec)} with {len(trainingLabels)} labels')
print(f'testing set is shape: {np.shape(testingSpec)} with {len(testingLabels)} labels')


# map between 0 and 1
maxValue = max(np.amax(trainingSpec), np.amax(trainingSpec))
minValue = min(np.amin(trainingSpec), np.amin(trainingSpec))

trainingSpec = (trainingSpec - minValue) / (maxValue - minValue)
testingSpec = (testingSpec - minValue) / (maxValue - minValue)


trainingDataset = tf.data.Dataset.from_tensor_slices((trainingSpec, trainingLabels))
testingDataset = tf.data.Dataset.from_tensor_slices((testingSpec, testingLabels))


del trainingSpec, trainingLabels, testingSpec, testingLabels

BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 100

trainingDataset = trainingDataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
testingDataset = testingDataset.batch(BATCH_SIZE)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(150, 320)),
    keras.layers.Dense(4000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(8, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainingDataset, epochs=10)

testLoss, testAcc = model.evaluate(testingDataset, verbose=2)






















