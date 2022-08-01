from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os
import sounddevice as sd
from pydub import AudioSegment
from PIL import Image
import time
from IPython import display


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})


def getMag(inputArray):
    if type(inputArray).__name__ != 'ndarray':
        inputArray = np.array([inputArray])

    # creates an empty list with the same shape as the input
    magnitude = np.emnp.pipty(inputArray.shape)
    # interates through the input list converting to a real magnitude
    for i in range(inputArray.size):
        magnitude[i] = (np.sqrt(inputArray[i].real ** 2 + inputArray[i].imag ** 2)) / len(inputArray)
    # outputs magnitude
    return magnitude


def getTopHalf(inputArray):
    # create empty list with half the length of the input list
    outArray = np.empty(int(inputArray.size/2))
    # Increment through the list doubling everything
    for i in range(int(inputArray.size/2)):
        outArray[i] = inputArray[int(i+(inputArray.size/2))]
    # output new list
    return outArray


def extend(arr, repLen):
    out = np.zeros(len(arr)+(repLen*2))
    for i in range(len(arr)):
        out[i+repLen] = arr[i]
    for i in range(repLen):
        out[len(arr)+i+repLen] = arr[-i-2]
        out[i] = arr[repLen-i]
    return out


def getGaussian(gaussianWidth, staDevCount=8, platformLength=0, staDev=1):
    out = np.arange(staDevCount * -0.5 * staDev, staDevCount * 0.5 * staDev, staDevCount * staDev / gaussianWidth)
    for x in range(gaussianWidth):
        out[x] = np.exp((-1*(out[x]**2))/(2*(staDev**2)))

    if platformLength != 0:
        platformOut = np.ones(len(out)+platformLength)
        for i in range(int(gaussianWidth / 2)):
            platformOut[i] = out[i]
            platformOut[-i-1] = out[-i-1]
        return platformOut
    else:
        return out


def getSpectrogram(audioArray, winSize, staDevCount, shiftSize, platformLength=0, padLength=0):
    gaussian = getGaussian(winSize - platformLength, staDevCount, platformLength=platformLength)
    newAudioArray = extend(audioArray, int(winSize/2))
    specImg = []
    for winCount in range(int(len(audioArray)/shiftSize)):
        windowArray = np.zeros(winSize)
        for sample in range(winSize):
            windowArray[sample] = newAudioArray[sample+(winCount*shiftSize)]
        windowArray = zeroPad(windowArray * gaussian, padLength, aliment='center')
        specImg.append(getTopHalf(getMag(np.fft.fftshift(np.fft.fft(windowArray)))))
    return np.swapaxes(np.array(specImg), 0, 1)


def downSample(arr, hz, newHz):
    newLen = int((newHz/hz)*len(arr))
    outArr = np.zeros(int(newLen))
    skipSize = int(hz/newHz)
    if skipSize != round(skipSize):
        print('weird downSample')

    for i in range(newLen):
        outArr[i]= arr[i*skipSize]

    return outArr


def gate(arr, closeThresh, threshRange, release):
    level = 0
    for i in range(len(arr)):
        if abs(arr[i]) > closeThresh+threshRange:
            level = abs(arr[i])
        else:
            level -= threshRange/release
        if level < closeThresh:
            arr[i] = 0

    return arr


def split(arr, splitCount, threshRange=500, closeThresh=1000, release=1000):
    level = 0
    starts = []
    ends = []
    maxes = []
    currentMax = 0
    out = []
    started = False
    level = 0
    for i in range(len(arr)):
        if abs(arr[i]) > closeThresh+threshRange:
            currentMax = max(arr[i], currentMax)
            level = abs(arr[i])
            if not started:
                started = True
                starts.append(i)

        else:
            level -= threshRange/release

        if level < closeThresh or i == len(arr)-1:
            arr[i] = 0
            if started:
                started = False
                ends.append(i)
                maxes.append(currentMax)
                currentMax = 0

    if splitCount != len(starts) or len(starts) != len(ends):
        print('oopsy woopsy')
        print(f'split cound: {splitCount}')
        print(f'starts: {len(starts)}')
        print(f'ends: {len(ends)}')

    for i in range(splitCount):
        tempOut = []
        for x in range(starts[i], ends[i]):
            tempOut.append(arr[x])
        out.append(tempOut)
    return out


def zeroPad(arr, addedLength, aliment='left', negAddedLengthReturnLength=12000):
    if addedLength >= 0:
        if aliment == 'right':
            out = np.zeros(int(len(arr)+addedLength))
            for i in range(len(arr)):
                out[-i-1] = arr[-i-1]

        elif aliment == 'left':
            out = np.zeros(int(len(arr)+addedLength))
            for i in range(len(arr)):
                out[i] = arr[i]

        elif aliment == 'center':
            out = zeroPad(arr, addedLength/2, aliment='right')
            for i in range(int(addedLength/2)):
                out = np.append(out, 0)
    else:
        out = np.zeros(negAddedLengthReturnLength)
        for i in range(len(out)):
            out[i] = arr[i]
    return out


def splitAndZeroPad(arr, splitCount, threshRange=500, closeThresh=1000, release=100, neededLen=24000):
    splitList = split(arr, splitCount, threshRange, closeThresh, release)
    for i in range(len(splitList)):
        splitList[i] = zeroPad(splitList[i], neededLen-len(splitList[i]))
    return splitList


def logForSpectrogram(arr, base):
    denominator = np.log(base)
    for x, col in enumerate(arr):
        for y, val in enumerate(col):
            arr[x, y] = max(np.log(val)/denominator, 0)

    return arr


dirList = os.listdir('/home/jonatank/Documents/inter/data/mini_speech_commands/')
# os.mkdir('/home/jonatank/Documents/inter/spec_data3/')
for folder, word in enumerate(dirList):
    # os.mkdir(f'/home/jonatank/Documents/inter/spec_data3/{word}/')
    inPath = '/home/jonatank/Documents/inter/data/mini_speech_commands/' + 'down'
    outPath = '/home/jonatank/Documents/inter/spec_data3/' + word

    fileList = os.listdir(inPath)

    for i, fname in enumerate(fileList):
        startTime = time.time()
        sampleRate, wav = wavfile.read(inPath+'/'+fname)
        wav = zeroPad(wav, 16000 - len(wav))
        # wav = gate(wav, np.amax(wav)//3, np.amax(wav)//5, 1000)
        # wav = np.array(splitAndZeroPad(wav, 1, np.amax(wav)//5, np.amax(wav)//3, np.amax(wav)//4, 12000)).flatten()
        spec = getSpectrogram(wav, 256, 6, 128, 220)

        print('Waveform shape:', np.shape(wav))
        print('Spectrogram shape:', np.shape(spec))
        fig, axs = plt.subplots(2)
        axs[0].plot(wav)
        axs[1].imshow(np.log2(spec), origin='lower', aspect='auto', extent=[0, 16000, 0, 8000])
        display.display(display.Audio(wav, rate=16000))
        plt.show()

        # np.save(f'{outPath}/{i}.npy', spec)
        endTime = time.time()
        print(f'{i}/{len(fileList)-1} in {folder}/{len(dirList)-1}  took: {round(endTime-startTime, 2)} seconds')




