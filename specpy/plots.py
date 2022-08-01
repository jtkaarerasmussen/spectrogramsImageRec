import numpy as np
import matplotlib.pyplot as plt
import os

labels = np.array(['go', 'yes', 'down', 'right', 'up', 'left', 'no', 'stop'])

trainingPath = '/home/jonatank/Documents/inter/spec3/training'
testingPath = '/home/jonatank/Documents/inter/spec3/testing'

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

for i in range(40):
    plt.imshow(np.log(trainingSpec[i]+np.finfo(float).eps), origin='lower')
    plt.title(trainingLabels[i])
    plt.show()