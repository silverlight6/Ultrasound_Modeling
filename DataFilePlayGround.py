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
    savePath = "/data/TBI/Datasets/NPFiles/"
    print("saved in : {}".format(savePath))
    np.save(savePath + "xAxis.npy", xaxis)
    np.save(savePath + "yAxis.npy", yaxis)


def FetchPolarCNNData(datapath):
    # load the data
    Harmonics = loadmat(datapath)

    # load each subsection needed
    harmonic = np.array(list(Harmonics['harmonics']))
    normalMask = np.array(list(Harmonics['normalMask']))
    bloodMask = np.array(list(Harmonics['bloodMask']))
    brainMask = np.array(list(Harmonics['brainMask']))
    bMode = np.array(list(Harmonics['bMode']))

    # separate real and imaginary.
    real = harmonic.real
    imag = harmonic.imag

    # real = real - real.mean(axis=0).mean(axis=0)
    # safe_max = np.abs(real).max(axis=0).max(axis=0)
    # safe_max[safe_max == 0] = 1
    # real = real / safe_max

    # imag = imag - imag.mean(axis=0).mean(axis=0)
    # safe_max = np.abs(imag).max(axis=0).max(axis=0)
    # safe_max[safe_max == 0] = 1
    # imag = imag / safe_max

    # turn nan to zero
    bloodMask = np.nan_to_num(bloodMask)
    normalMask = np.nan_to_num(normalMask)
    bMode = np.nan_to_num(bMode)
    # bMode = bMode.mean(axis=-1)
    # print(bMode.shape)

    # delete non-brain from input data
    # for i in range(0, 7):
    #     real[:, :, i] = np.where(brainMask == 0, 0, real[:, :, i])
    #     imag[:, :, i] = np.where(brainMask == 0, 0, imag[:, :, i])

    # bMode = np.where(brainMask == 0, 0, bMode)

    label = np.where(bloodMask > normalMask, 1, 0)
    # label = np.where(brainMask == 0, 0, label)

    label = label.astype('float32')
    real = real.astype('float64')
    imag = imag.astype('float64')
    bMode = bMode.astype('float64')
    label = cv2.resize(label, (64, 256), interpolation=cv2.INTER_CUBIC)
    real = cv2.resize(real, (64, 256), interpolation=cv2.INTER_CUBIC)
    imag = cv2.resize(imag, (64, 256), interpolation=cv2.INTER_CUBIC)
    bMode = cv2.resize(bMode, (64, 256), interpolation=cv2.INTER_CUBIC)
    # tOutput = cv2.resize(tOutput, (64, 256), interpolation=cv2.INTER_CUBIC)
    bMode = bMode.reshape(256, 64, 1)
    label = label.reshape(256, 64, 1)

    # # Debugging code below

    # temphist = label.flatten()
    # # Create histogram
    # plt.hist(temphist, bins=10, histtype='bar', log=True)
    # # configure and draw the histogram figure
    # plt.title("Label Value Histogram")
    # plt.xlabel("Label Value")
    # plt.ylabel("Number per Value")
    # plt.show()

    # for i in range(0, 7):
    #     plt.figure(figsize=(10, 10))
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(real[:, :, i], cmap='autumn')
    #     plt.show()
    #
    # for i in range(0, 7):
    #     plt.figure(figsize=(10, 10))
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(imag[:, :, i], cmap='autumn')
    #     plt.show()
    #
    # for i in range(0, 3):
    #     plt.figure(figsize=(10, 10))
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(tOutput[:, :, i], cmap='autumn')
    #     plt.show()

    # tempLabel = np.where(label1 > 0, 1, 0)
    # tempLabel = tempLabel + label2
    # tempLabel = np.where(label2 == 1, tempLabel + 1, tempLabel)
    # plt.figure(figsize=(10, 10))
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(False)
    # plt.imshow(tempLabel, cmap='winter')
    # plt.show()

    # concatenate the columns into one structure
    combinedData = np.concatenate((label, real, imag, bMode), axis=2)

    # print("in fetch data - {}".format(combinedData.shape))

    return combinedData


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
    change = False
    if r % 3:
        image = np.fliplr(image)
        change = True
    # if r % 5:
    #     image = ndimage.zoom(image, .9 + ((r % 6) / 25))
    #     change = True
    if r % 7:
        image = ndimage.rotate(image, ((r % 6) / 50 - .05), reshape=False)
        change = True
    if not change:
        image[:, :, 1:] = noisy(image[:, :, 1:], r)
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


def output2DImages():
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

    IPH_patients = [8, 9, 10, 47, 53, 62, 66, 67, 69, 74, 75, 78, 85, 89, 101]
    # counting files
    count = 0

    def fileLoop(path, patient_num, iterator):
        for harmonic in os.listdir(path):
            if ".mat" in harmonic:
                # print("Checkpoint 1")
                hPath = os.path.join(path, harmonic)                  # put file name on path
                pathName = harmonic[0:17]                             # get path to print
                image = FetchPolarCNNData(hPath)
                # print(image.shape)
                iLabel_num = np.count_nonzero(image[:, :, 0] == 2) / (256 * 64)
                # if iLabel_num > .015:
                #     print(iLabel_num)
                iLabel = iLabel_num > .01
                lock = multiprocessing.Lock()
                lock.acquire()
                if count % 10 == iterator:                                        # 44 patients in training
                    validationData.append([image])
                    validationPaths.append([pathName])
                    validationLabels.append([iLabel])
                elif count % 10 + 1 == iterator or count % 10 - 9 == iterator:                                      # 4 patients in testing
                    testingData.append([image])
                    testingPaths.append([pathName])
                    testingLabels.append([iLabel])
                else:                                                 # everything else in validation
                    trainingData.append([image])
                    trainingPaths.append([pathName])
                    trainingLabels.append([iLabel])
                    # for _ in range(0, math.floor(iLabel_num / .15)):
                    #     trainingData.append([image])
                    #     trainingPaths.append([pathName])
                    #     trainingLabels.append([iLabel])
                    # if patient_num in IPH_patients:
                    #     for i in range(0, 4):
                    #         image = imageReduc(image)
                    #         trainingData.append([image])
                    #         trainingPaths.append([pathName])
                    #         trainingLabels.append([iLabel])
                lock.release()
                # for testing purposes only
        print(count)
        # print("in fileloop - {}".format(trainingData))
        return

    # data paths; it is just what it sounds like
    # watch out for polar versus non-polar
    dataPaths = '/data/TBI/Datasets/PolarData/'

    # make sure that data gets looked at even

    processes = []
    pathlist = os.listdir(dataPaths)
    pathlist = np.sort(pathlist)
    pathlist = shuffle(pathlist, random_state=20)
    for path in pathlist:
        num = path[3:]
        num = int(num)
        fpath = os.path.join(dataPaths, path)
        p = multiprocessing.Process(target=fileLoop, args=(fpath, num))
        p.start()
        processes.append(p)
        count += 1

    for process in processes:
        print(count)
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
    # let us see what we got
    print(testingPaths)
    print("training {}".format(trainingData.shape))
    print("testing {}".format(testingData.shape))
    print("validation {}".format(validationData.shape))
    print("Positive # label {}".format(np.count_nonzero(trainingLabels)))

    savePath = "/data/TBI/Datasets/NPFiles/Polar2Class/"
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


output2DImages()
# FetchPolarAxis('/data/TBI/Datasets/PolarData/DoD001/DoD001_Ter001_RC1_Harmonics_Polar.mat')
# FetchTimeData('/data/TBI/Datasets/PolarData/DoD001/DoD001_Ter001_RC1_Harmonics_Polar.mat')
