import numpy as np
import os                           # reading files
import cv2                          # resize
import math
import cmath
import multiprocessing
import random
import scipy.ndimage as ndimage     # rotate, zoom
from sklearn.utils import shuffle   # shuffle seed


from matplotlib import pyplot as plt
from scipy.io import loadmat


def FetchTimeData(datapath):
    Harmonics = loadmat(datapath)
    harmonic = np.array(list(Harmonics['harmonics']))

    harmShape = harmonic.shape
    period = 50         # number of time-ticks along the waveform,
    tt = np.linspace(1, period, period)
    form = np.zeros([harmShape[0], harmShape[1], period])  # allocate memory to waveform.

    mag = np.sqrt(np.square(harmonic.real) + np.square(harmonic.imag))

    for j in range(1, harmShape[1]):
        for i in range(1, harmShape[0]):
            for k in range(1, 7):
                phase = cmath.phase(harmonic[i, j, k])
                # the harmonics here are the magnitude and phase, not the real and imaginary parts.
                form[i, j, :] = form[i, j, :] + mag[i, j, k] * np.sin(2 * k * tt * math.pi / period + phase)

    normalMask = np.array(list(Harmonics['normalMask']))
    bloodMask = np.array(list(Harmonics['bloodMask']))
    brainMask = np.array(list(Harmonics['brainMask']))

    bloodMask = np.nan_to_num(bloodMask)
    normalMask = np.nan_to_num(normalMask)

    label = np.where(bloodMask > normalMask, 2, 1)
    label = np.where(brainMask == 0, 0, label)

    real = harmonic.real
    imag = harmonic.imag

    M1 = np.sqrt(np.square(real[:, :,  0]) + np.square(imag[:, :, 0]))
    MO = np.sqrt(np.square(real[:, :,  0]) + np.square(imag[:, :, 0]))

    for i in range(1, 7):
        MO += np.sqrt(np.square(real[:, :, i]) + np.square(imag[:, :, i]))

    M1 = M1 / MO
    tOutput = np.zeros([form.shape[0], form.shape[1], 3])

    tOutput[:, :, 0] = form[:, :, 0]
    tOutput[:, :, 1] = form[:, :, 17]
    tOutput[:, :, 2] = M1

    tOutput = tOutput - tOutput.mean(axis=0).mean(axis=0)
    safe_max = np.abs(tOutput).max(axis=0).max(axis=0)
    safe_max[safe_max == 0] = 1
    tOutput = tOutput / safe_max

    for i in range(0, 3):
        tOutput[:, :, i] = np.where(brainMask == 0, 0, tOutput[:, :, i])

    tOutput = cv2.resize(tOutput, (64, 256), interpolation=cv2.INTER_CUBIC)

    label = label.astype('float32')
    label = cv2.resize(label, (64, 256), interpolation=cv2.INTER_CUBIC)

    label = label.reshape(256, 64, 1)
    tOutput = tOutput.reshape(256, 64, 3)

    combinedData = np.concatenate((label, tOutput), axis=-1)

    return combinedData


def FetchPolarAxis(datapath):
    Harmonics = loadmat(datapath)
    print(Harmonics.keys())

    xaxis = np.array(list(Harmonics['xAxis']))
    yaxis = np.array(list(Harmonics['zAxis']))

    xaxis = cv2.resize(xaxis, (64, 256), interpolation=cv2.INTER_AREA)
    yaxis = cv2.resize(yaxis, (64, 256), interpolation=cv2.INTER_AREA)

    xaxis += 100
    yaxis -= 4

    home = os.path.expanduser("~")
    savePath = "/DATA/TBI/Datasets/NPFiles/"
    print("saved in : {}".format(savePath))
    np.save(savePath + "xAxis.npy", xaxis)
    np.save(savePath + "yAxis.npy", yaxis)


def noisy(image, r):
    row, col, ch = image.shape
    mean = 0
    var = 0.001 + (r % 20 / 1000)
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape([row, col, ch])
    noisy = image + gauss
    return noisy


def dataAug(image):
    r = random.randint(0, 100000)
    if r % 3:
        image = np.fliplr(image)
    # if r % 5:
    #     image = ndimage.zoom(input=image, zoom=.9 + ((r % 6) / 25), output=image, mode='nearest')
    if r % 7:
        image = ndimage.rotate(image, ((r % 6) / 50 - .05), reshape=False)
        print(image[:, :, 0])
    if r % 11:
        image[:, :, 1:] = noisy(image[:, :, 1:], r)
        print(image[:, :, 0])
    return image


def imageReduc(image):
    output = image
    si = image.shape
    # print("si = {}".format(si))
    mask = np.where(image[:, :, 0] == 0, 1, 0)
    mask2 = np.zeros((si[0], si[1]), dtype=np.int32)
    for _ in range(0, 2):                   # shrinks input by 2 pixels
        for i in range(1, si[0] - 1):
            for j in range(1, si[1] - 1):
                if mask[i, j] == 1:
                    mask2[i - 1, j] = 1
                    mask2[i, j - 1] = 1
                    mask2[i + 1, j] = 1
                    mask2[i, j + 1] = 1
                    mask2[i - 1, j - 1] = 1
                    mask2[i - 1, j + 1] = 1
                    mask2[i + 1, j + 1] = 1
                    mask2[i + 1, j - 1] = 1
                    mask2[i, j] = 1
        mask = mask2
        mask2 = np.zeros((si[0], si[1]), dtype=np.int32)
    output[:, :, 0] = np.where(mask == 1, 0, output[:, :, 0])
    for k in range(1, si[2]):
        output[:, :, k] = np.where(output[:, :, 0] == 0, 0, output[:, :, k])

    return output


def output2DImages(iteration):
    # Make arrays for dividing data between training, testing, and validation.
    manager = multiprocessing.Manager()

    trainingData = manager.list()
    testingData = manager.list()
    validationData = manager.list()
    trainingPaths = manager.list()
    testingPaths = manager.list()
    validationPaths = manager.list()
    trainingLabels = manager.list()
    testingLabels = manager.list()
    validationLabels = manager.list()
    trainingCycles = manager.list()
    testingCycles = manager.list()
    validationCycles = manager.list()

    IPH_patients = [8, 9, 10, 47, 53, 62, 66, 67, 69, 74, 75, 78, 85, 89, 101]
    # counting files
    count = 0

    def fileLoop(path, iteration):
        iteration = iteration % 10
        for harmonic in os.listdir(path):
            if ".mat" in harmonic:
                # print("Checkpoint 1")
                hPath = os.path.join(path, harmonic)                  # put file name on path
                pathName = harmonic[0:17]                             # get path to print
                Harmonics = loadmat(hPath)
                # print(Harmonics.keys())

                # load each subsection needed
                harmonic = np.array(list(Harmonics['harmonics']))
                normalMask = np.array(list(Harmonics['normalMask']))
                bloodMask = np.array(list(Harmonics['bloodMask']))
                brainMask = np.array(list(Harmonics['brainMask']))
                bMode = np.array(list(Harmonics['bModeNorm']))

                # separate real and imaginary.
                real = harmonic.real
                imag = harmonic.imag

                # turn nan to zero
                bloodMask = np.nan_to_num(bloodMask)
                normalMask = np.nan_to_num(normalMask)
                bMode = np.nan_to_num(bMode)
                # bMode = bMode.mean(axis=-1)
                bMode = np.log2(bMode)

                label = np.where(bloodMask > normalMask, 2, 1)
                label = np.where(brainMask == 0, 0, label)

                label = label.astype('float32')
                label = cv2.resize(label, (64, 256))
                label = label.reshape(256, 64, 1)

                cycles = real.shape[-1]

                real = real.astype('float64')
                imag = imag.astype('float64')
                bMode = bMode.astype('float64')
                iLabel_num = np.count_nonzero(label[:, :, 0] == 2) / (256 * 64)
                iLabel = iLabel_num > .01
                patient_num = pathName[4:6]
                patient_num = int(patient_num)
                # print(pathName)
                # print(patient_num)
                # print(iLabel_num)

                for k in range(0, cycles):
                    realO = real[:, :, :, k]
                    imagO = imag[:, :, :, k]
                    bModeO = bMode[:, :, 0, k]
                    # delete non-brain from input data
                    for i in range(0, 7):
                        realO[:, :, i] = np.where(brainMask == 0, 0, realO[:, :, i])
                        imagO[:, :, i] = np.where(brainMask == 0, 0, imagO[:, :, i])

                    # bModeO = np.where(brainMask == 0, 0, bModeO)
                    realO = cv2.resize(realO, (64, 256))
                    imagO = cv2.resize(imagO, (64, 256))
                    bModeO = cv2.resize(bModeO, (64, 256))
                    # tOutput = cv2.resize(tOutput, (64, 256), interpolation=cv2.INTER_CUBIC)

                    bModeO = bModeO.reshape([256, 64, 1])

                    # concatenate the columns into one structure
                    image = np.concatenate((label, realO, imagO, bModeO), axis=2)
                    # print("appendData.shape = {}".format(appendData.shape))

                    lock = multiprocessing.Lock()
                    lock.acquire()
                    if count % 10 == iteration:                                        # 44 patients in training
                        testingData.append([image])
                        testingPaths.append([pathName])
                        testingLabels.append([iLabel])
                        testingCycles.append([cycles])
                    elif count % 10 == iteration + 1:                                                 # everything else in validation
                        validationData.append([image])
                        validationPaths.append([pathName])
                        validationLabels.append([iLabel])
                        validationCycles.append([cycles])
                    else:
                        trainingData.append([image])
                        trainingPaths.append([pathName])
                        trainingLabels.append([iLabel])
                        trainingCycles.append([cycles])
                        for y in range(0, math.floor(iLabel_num / .02)):
                            pathNameB = pathName + "_{}".format(y)
                            print(pathNameB)
                            trainingData.append([image])
                            trainingPaths.append([pathNameB])
                            trainingLabels.append([iLabel])
                            trainingCycles.append([cycles])
                        if patient_num in IPH_patients:
                            for x in range(0, 4):
                                lock.release()
                                image = imageReduc(image)
                                pathName_iph = pathName + "iph_{}".format(x)
                                print(pathName_iph)
                                lock.acquire()
                                trainingData.append([image])
                                trainingPaths.append([pathName_iph])
                                trainingLabels.append([iLabel])
                                trainingCycles.append([cycles])
                        # for _ in range(0, 2):
                        #     lock.release()
                        #     image = dataAug(image)
                        #     lock.acquire()
                        #     trainingData.append([image])
                        #     trainingPaths.append([pathName])
                        #     trainingLabels.append([iLabel])
                        #     trainingCycles.append([cycles])
                    lock.release()
                # for testing purposes only
        print(count)
        # print("in fileloop - {}".format(trainingData))
        return

    # data paths; it is just what it sounds like
    # watch out for polar versus non-polar
    dataPaths = '/DATA/TBI/Datasets/CardiacData/'

    # make sure that data gets looked at even

    processes = []
    pathlist = os.listdir(dataPaths)
    pathlist = np.sort(pathlist)
    pathlist = shuffle(pathlist, random_state=20)
    for path in pathlist:
        fpath = os.path.join(dataPaths, path)
        p = multiprocessing.Process(target=fileLoop, args=(fpath, iteration))
        p.start()
        processes.append(p)
        count += 1

    for process in processes:
        # print("in process loop - {}".format(trainingData))
        process.join()

    # convert to numpy arrays, because the data is 4D
    trainingData = np.array(trainingData)
    testingData = np.array(testingData)
    validationData = np.array(validationData)
    trainingPaths = np.array(trainingPaths)
    testingPaths = np.array(testingPaths)
    validationPaths = np.array(validationPaths)
    trainingLabels = np.array(trainingLabels)
    testingLabels = np.array(testingLabels)
    validationLabels = np.array(validationLabels)
    trainingCycles = np.array(trainingCycles)
    testingCycles = np.array(testingCycles)
    validationCycles = np.array(validationCycles)

    # let us see what we got
    print(testingPaths)
    print("training {}".format(trainingData.shape))
    print("testing {}".format(testingData.shape))
    print("validation {}".format(validationData.shape))
    print("Cycle count train {}".format(trainingCycles.shape))
    print("Cycle count test {}".format(testingCycles.shape))
    print("Cycle count validation {}".format(validationCycles.shape))
    print("Positive # label {}".format(np.count_nonzero(trainingLabels)))

    savePath = "/DATA/TBI/Datasets/NPFiles/CardiacBalanced/"
    print("saved in : {}".format(savePath))
    np.save(savePath + "TrainingData.npy", trainingData)
    np.save(savePath + "TestingData.npy", testingData)
    np.save(savePath + "ValidationData.npy", validationData)
    np.save(savePath + "TrainingPaths.npy", trainingPaths)
    np.save(savePath + "TestingPaths.npy", testingPaths)
    np.save(savePath + "ValidationPaths.npy", validationPaths)
    np.save(savePath + "TrainingLabels.npy", trainingLabels)
    np.save(savePath + "TestingLabels.npy", testingLabels)
    np.save(savePath + "ValidationLabels.npy", validationLabels)
    np.save(savePath + "TrainingCycles.npy", trainingCycles)
    np.save(savePath + "TestingCycles.npy", testingCycles)
    np.save(savePath + "ValidationCycles.npy", validationCycles)



if __name__ == '__main__':

    output2DImages(7)
# FetchPolarAxis('/data/TBI/Datasets/PolarData/DoD001/DoD001_Ter001_RC1_Harmonics_Polar.mat')
# FetchTimeData('/data/TBI/Datasets/PolarData/DoD001/DoD001_Ter001_RC1_Harmonics_Polar.mat')
