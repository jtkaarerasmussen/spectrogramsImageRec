import numpy as np
import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})


def getMag(inputArray):
    if type(inputArray).__name__ != 'ndarray':
        inputArray = np.array([inputArray])

    # creates an empty list with the same shape as the input
    magnitude = np.empty(inputArray.shape)
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
        out[repLen+len(arr)-1+i] = arr[len(arr)-i-1]
        out[repLen-i] = arr[i]
    return out


t = np.arange(0, 1, 0.001)
f = np.arange(1/0.001)
wav = np.sin(t*2*np.pi)
t = np.arange(0, 3, 0.001)

wav = extend(wav,1000)
# for i in range(int(len(wav)/2)):
#     wav[int(i+(len(wav)/2))] = 0
plt.plot(wav)
plt.show()
plt.plot(getTopHalf(getMag(np.fft.fftshift(np.fft.fft(wav)))))
plt.show()


