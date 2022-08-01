import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os

labels = np.array(['go', 'yes', 'down', 'right', 'up', 'left', 'no', 'stop'])

trainingPath = '/home/jonatank/Documents/inter/spec3/training'
testingPath = '/home/jonatank/Documents/inter/spec3/testing'
valPath = '/home/jonatank/Documents/inter/spec3/validation'

trainingDirList = os.listdir(trainingPath)
testingDirList = os.listdir(testingPath)


# load training
trainingSpec = []
trainingLabels = []

for dirName in trainingDirList:
    tempFileList = os.listdir(f'{trainingPath}/{dirName}')
    for fname in tempFileList:
        loaded = np.load(f'{trainingPath}/{dirName}/{fname}').astype(np.single)
        loaded = loaded[..., np.newaxis]
        trainingSpec.append(loaded)
        trainingLabels.append(np.where(labels == dirName))

trainingSpec = np.array(trainingSpec)
trainingLabels = np.array(trainingLabels).flatten()


# load testing
testingSpec = []
testingLabels = []

for dirName in testingDirList:
    tempFileList = os.listdir(f'{testingPath}/{dirName}')
    for fname in tempFileList:
        loaded = np.load(f'{testingPath}/{dirName}/{fname}').astype(np.single)
        loaded = loaded[..., np.newaxis]
        testingSpec.append(loaded)
        testingLabels.append(np.where(labels == dirName))

testingSpec = np.array(testingSpec)
testingLabels = np.array(testingLabels).flatten()


# load validation
valSpec = []
valLabels = []

for dirName in testingDirList:
    tempFileList = os.listdir(f'{valPath}/{dirName}')
    for fname in tempFileList:
        loaded = np.load(f'{valPath}/{dirName}/{fname}').astype(np.single)
        loaded = loaded[..., np.newaxis]
        valSpec.append(loaded)
        valLabels.append(np.where(labels == dirName))

valSpec = np.array(testingSpec)
valLabels = np.array(testingLabels).flatten()

print(f'training set is shape: {np.shape(trainingSpec)} with {len(trainingLabels)} labels')
print(f'testing set is shape: {np.shape(testingSpec)} with {len(testingLabels)} labels')
print(f'validation set is shape: {np.shape(valSpec)} with {len(valLabels)} labels')

# map between 0 and 1
maxValue = max(np.amax(trainingSpec), np.amax(trainingSpec), np.amax(valSpec))
minValue = min(np.amin(trainingSpec), np.amin(trainingSpec), np.min(valSpec))

trainingSpec = (trainingSpec - minValue) / (maxValue - minValue)
testingSpec = (testingSpec - minValue) / (maxValue - minValue)
valSpec = (valSpec - minValue) / (maxValue - minValue)

trainingDataset = tf.data.Dataset.from_tensor_slices((trainingSpec, trainingLabels))
testingDataset = tf.data.Dataset.from_tensor_slices((testingSpec, testingLabels))
valDataset = tf.data.Dataset.from_tensor_slices((valSpec, valLabels))

del trainingSpec, trainingLabels, testingSpec, testingLabels, valSpec, valLabels


for spectrogram, _ in trainingDataset.take(1):
    inputShape = spectrogram.shape
print('input shape:', inputShape)


BATCH_SIZE = 64
trainingDataset = trainingDataset.shuffle(BATCH_SIZE)
trainingDataset = trainingDataset.batch(BATCH_SIZE)
testingDataset = testingDataset.batch(BATCH_SIZE)
valDataset = valDataset.batch(BATCH_SIZE)

normLayer = keras.layers.Normalization()
normLayer.adapt(data=trainingDataset.map(map_func=lambda spec, label: spec))


# model = keras.Sequential([
#     keras.layers.Input(shape=inputShape),
#     keras.layers.Resizing(32, 32),
#     normLayer,
#     keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Dropout(0.25),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(8, activation='softmax')
# ])

model = tf.keras.models.load_model('models/8')


model.summary()

opt = keras.optimizers.Adadelta()
EPOCHS = 100

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainingDataset, epochs=EPOCHS, validation_data=valDataset, callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))

testLoss, testAcc = model.evaluate(testingDataset, verbose=2)

model.save('models/9')

metrics = history.history

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('loss and accuracy')
ax1.plot(history.epoch, metrics['loss'])
ax1.plot(history.epoch, metrics['val_loss'])
ax2.plot(history.epoch, metrics['val_loss'])
ax2.plot(history.epoch, metrics['val_accuracy'])
plt.show()





















