import numpy as np
import config
import os                           # reading files
import random
import cv2                          # resize
import math
import cmath
import multiprocessing
import random
import scipy.ndimage as ndimage     # rotate, zoom
import time
import matplotlib
from sklearn.utils import shuffle   # shuffle seed

from matplotlib import pyplot as plt
from scipy.io import loadmat

matplotlib.use('Agg')


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

    tOutput = cv2.resize(tOutput, (80, 256), interpolation=cv2.INTER_CUBIC)

    label = label.astype('float32')
    label = cv2.resize(label, (80, 256), interpolation=cv2.INTER_CUBIC)

    label = label.reshape(256, 80, 1)
    tOutput = tOutput.reshape(256, 80, 3)

    combinedData = np.concatenate((label, tOutput), axis=-1)

    return combinedData


def FetchPolarAxis(datapath, axisPath):
    Harmonics = loadmat(datapath)
    print(Harmonics.keys())

    xaxis = np.array(list(Harmonics['xAxis']))
    yaxis = np.array(list(Harmonics['zAxis']))

    xaxis = cv2.resize(xaxis, (80, 256), interpolation=cv2.INTER_AREA)
    yaxis = cv2.resize(yaxis, (80, 256), interpolation=cv2.INTER_AREA)

    xaxis += 100
    yaxis -= 4

    print("saved in : {}".format(axisPath))
    np.save(axisPath + "xAxis.npy", xaxis)
    np.save(axisPath + "yAxis.npy", yaxis)


def shift(image):
    r = random.randint(0, 30)
    c = random.randint(0, 12)
    direction = random.randint(0, 1)
    si = image.shape
    mask2 = np.zeros((si[0], si[1], si[2]), dtype=np.float64)
    for i in range(0, si[0] - 1):
        for j in range(0, si[1] - 1):
            if direction:
                if 0 <= i + r < si[0] and 0 <= j + c < si[1]:
                    mask2[i, j, :] = image[i + r, j + c, :]
            else:
                if 0 <= i - r < si[0] and 0 <= j - c < si[1]:
                    mask2[i, j, :] = image[i - r, j - c, :]
    return mask2


def clip(image):
    r = random.randint(0, 256)
    c = random.randint(0, 80)
    ra = random.randint(20, 40)
    ca = random.randint(10, 20)
    si = image.shape
    # print(si)
    for i in range(0, si[0] - 1):
        for j in range(0, si[1] - 1):
            if r + ra > i > r - ra and c + ca > j > c - ca:
                image[i, j, :] = 0
    return image


def noisy(image, r):
    row, col, ch = image.shape
    mean = 0
    # var = 0.001 + (r % 20 / 1000)
    var = 1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape([row, col, ch])
    gauss /= 5000
    noisy = image + gauss
    return noisy


def imageReduc(image):
    output = image
    si = image.shape
    # Where it is outside the brain, mark 1
    mask = np.where(image[:, :, 0] < 0.1, 1, 0)
    # Initialize to 0s
    mask2 = np.zeros((si[0], si[1]), dtype=np.int32)
    for _ in range(0, 2):                   # shrinks input by 2 pixels
        for i in range(1, si[0] - 1):
            for j in range(1, si[1] - 1):
                if mask[i, j] > 1:
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
    # shrink the mask by 2
    output[:, :, 0] = np.where(mask == 1, 0, output[:, :, 0])
    for k in range(1, si[2]):
        output[:, :, k] = np.where(output[:, :, 0] == 0, 0, output[:, :, k])
    print(output.shape)
    return output


def dataAug(image):
    r = random.randint(0, 100000)
    t = random.randint(0, 100000)
    # flip horizontal
    if r % 2:
        image = np.fliplr(image)
        image = np.array(image)
        # print("Shape after flip {}".format(image.shape))

    # flip vertical. Optional because it isn't possible in practice.
    # if t % 2:
    #     image = np.flipud(image)
    #     image = np.array(image)
        # print("Shape after flip {}".format(image.shape))

    if r % 3:
        image[:, :, :-1] = imageReduc(image[:, :, :-1])
        # print("Shape after reduc {}".format(image.shape))

    # Contrast
    if t % 3:
        # Create histogram
        brainMask = image[:, :, 0]

        # If not using norm, use this
        image[:, :, 1:] = np.max(image[:, :, 1:], keepdims=True) * (image[:, :, 1:] - np.min(image[:, :, 1:],
                          keepdims=True)) / (np.max(image[:, :, 1:], keepdims=True) - np.min(image[:, :, 1:], keepdims=True))
        # With norm, use this
        # image[:, :, 1:] = image[:, :, 1:] - np.min(image[:, :, 1:], keepdims=True) /
        # (np.max(image[:, :, 1:], keepdims=True) - np.min(image[:, :, 1:], keepdims=True))
        minval = np.percentile(image, 2)
        maxval = np.percentile(image, 98)
        image[:, :, 1:] = np.clip(image[:, :, 1:], minval, maxval)
        # if no norm, use this
        image[:, :, 1:] = maxval * (image[:, :, 1:] - minval) / (maxval - minval)
        # if norm use this
        # image[:, :, 1:] = (image[:, :, 1:] - minval) / (maxval - minval)
        for i in range(1, image.shape[-1]):
            image[:, :, i] = np.where(brainMask < 0.1, 0.0, image[:, :, i])

    # Zoom is simply broken. It is doing a total of 0 currently
    # if r % 5:
    #     dim = (image.shape[1], image.shape[0])
    #     image = cv2.resize(image, dim, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    #     for i in range(1, image.shape[-1]):
    #         image[:, :, i] = np.where(image[:, :, 0] < 0.1, 0.0, image[:, :, i])

    if r % 5:
        image = clip(image)

    # Rotate is not realistic for our application plus it tends to look poor.
    if r % 13:
        plt.show()
        image = ndimage.rotate(image, (r % 11) / 5, reshape=False)
        image = np.array(image)
        # print("Shape after rotate {}".format(image.shape))

    if t % 2:
        image = shift(image)
    if t % 5:
        image[:, :, 1:] = noisy(image[:, :, 1:], r)
        image = np.array(image)
        # print("Shape after noise {}".format(image.shape))

    # plt.figure(figsize=(10, 10))
    # plt.grid(False)
    # plt.imshow(image[:, :, 1], cmap='magma')
    # plt.title("Label After Augmentation")
    # plt.show()
    return image

 # objective indicates if the data for finding brain mask or bleed (0=brain, 1=bleed)
def output2DImages(iteration, objective, rawDataPath, savePath):
    # Make arrays for dividing data between training, testing, and validation.
    manager = multiprocessing.Manager()

    trainingData = manager.list()
    testingData = manager.list()
    # validationData = manager.list()
    trainingPaths = manager.list()
    testingPaths = manager.list()
    # validationPaths = manager.list()

    IPH_patients = [8, 9, 10, 12, 24, 47, 53, 62, 66, 67, 69, 74, 75, 78, 85, 89, 93,
                    101, 105, 107, 110, 112, 113, 120, 121, 126, 129, 130, 133]
    bad_patients = [1, 14, 22, 23, 27, 28, 32, 34, 35, 36, 37, 38, 39, 44, 49, 69, 71, 78, 82, 90, 98, 101, 121, 124,
                    128, 133, 136, 928]
    timeStart = np.zeros([100])
    timeEnd = np.zeros([100])
    # counting files
    count = 0

    # objective indicates if the data for finding brain mask or bleed (0=brain, 1=bleed)
    def fileLoop(path, patient_num, iteration, mode, objective):
        iteration = iteration % 10
        for file in os.listdir(path):
            if ".mat" in file:
                hPath = os.path.join(path, file)                  # put file name on path
                pathName = file[0:17]                             # get path to print
                # timeA = time.time()
                # print("Loading file {}".format(pathName))
                Harmonics = loadmat(hPath)
                # print("Finished file {} at time {}".format(pathName, time.time() - timeA))

                normalMask = np.array(list(Harmonics['normalMask']))
                bloodMask = np.array(list(Harmonics['bloodMask']))
                brainMask = np.array(list(Harmonics['brainMask']))
                bMode = np.array(list(Harmonics['bModeNorm']))
                # print("Keys are {}".format(Harmonics.keys()))

                if mode == 0:
                    harmonic = np.array(list(Harmonics['harmonics']))
                    # separate real and imaginary.
                    real = harmonic.real
                    imag = harmonic.imag
                else:
                    displacement = np.array(list(Harmonics['displacement']))
                    hrTimes = np.array(list(Harmonics['hrTimes']))
                    hrshape = hrTimes.shape
                    disshape = displacement.shape
                    real = np.zeros([disshape[0], disshape[1], 5, hrshape[1] - 1])
                    imag = np.zeros([disshape[0], disshape[1], 5, hrshape[1] - 1])
                    for h in range(0, hrshape[1] - 1):
                        start = int(math.ceil(30 * hrTimes[0, h]))
                        real[:, :, :, h] = displacement[:, :, start:start + 5]
                        imag[:, :, :, h] = displacement[:, :, start + 5:start + 10]
                    real = np.array(real)
                    imag = np.array(imag)

                # turn nan to zero
                # bloodMask = np.nan_to_num(bloodMask)
                # normalMask = np.nan_to_num(normalMask)
                # bMode = np.nan_to_num(bMode)
                # bMode = bMode.mean(axis=-1)
                bMode = np.log10(bMode)

                # Smooth & Create the Label
                brainMask = cv2.resize(brainMask, (80, 256))
                if (objective == 0):
                    # This code is for finding the brain mask
                    label = np.where(brainMask == 0, 0, 1)
                else:
                    # Finding the blood, brain, and outside brain masks
                    label = np.where(bloodMask > normalMask, 2, 1)
                    label = label.astype('float32')
                    label = cv2.GaussianBlur(src=label, ksize=(9, 9), sigmaX=4)
                    label = np.where(bloodMask > normalMask, 2, label)
                    label = cv2.GaussianBlur(src=label, ksize=(3, 3), sigmaX=2)
                    label = np.where(bloodMask > normalMask, 2, label)
                    label = cv2.resize(label, (80, 256))
                    label = np.where(brainMask == 0, 0, label)
                
                label = label.reshape([256, 80, 1])
                cycles = real.shape[-1]
                real = real.astype('float64')
                imag = imag.astype('float64')
                bMode = bMode.astype('float64')
                # iLabel_num = np.count_nonzero(label[:, :, 0] == 2) / (256 * 64)
                bMode = np.mean(bMode, axis=2)

                for k in range(0, cycles):
                    realO = real[:, :, :, k]
                    imagO = imag[:, :, :, k]
                    bModeO = bMode[:, :, k]

                    # print(bModeO.shape)

                    # realO = realO - np.expand_dims(realO.mean(axis=2), axis=2)
                    # rMax = np.expand_dims(np.abs(realO).max(axis=2), axis=2)
                    # realO = realO / (2 * rMax)
                    #
                    # imagO = imagO - np.expand_dims(imagO.mean(axis=2), axis=2)
                    # iMax = np.expand_dims(np.abs(imagO).max(axis=2), axis=2)
                    # imagO = imagO / (2 * iMax)

                    realO = realO - realO.mean(axis=0).mean(axis=0)
                    safe_max = np.abs(realO).max(axis=0).max(axis=0)
                    safe_max[safe_max == 0] = 1
                    realO = realO / safe_max

                    imagO = imagO - imagO.mean(axis=0).mean(axis=0)
                    safe_max = np.abs(imagO).max(axis=0).max(axis=0)
                    safe_max[safe_max == 0] = 1
                    imagO = imagO / safe_max

                    # bModeO = np.where(brainMask == 0, 0, bModeO)
                    realO = cv2.resize(realO, (80, 256))
                    imagO = cv2.resize(imagO, (80, 256))
                    bModeO = cv2.resize(bModeO, (80, 256))
                    # tOutput = cv2.resize(tOutput, (80, 256), interpolation=cv2.INTER_CUBIC)

                    if (objective == 1):
                        # delete non-brain from input data
                        for i in range(0, realO.shape[-1]):
                            realO[:, :, i] = np.where(brainMask == 0, 0.0, realO[:, :, i])
                            imagO[:, :, i] = np.where(brainMask == 0, 0.0, imagO[:, :, i])

                    bModeO = bModeO.reshape([256, 80, 1])

                    # concatenate the columns into one structure
                    image = np.concatenate((label, realO, imagO, bModeO), axis=2)

                    lock = multiprocessing.Lock()
                    lock.acquire()
                    if count % 10 == iteration:
                        testingData.append([image])
                        testingPaths.append([pathName])
                    else:
                        trainingData.append([image])
                        trainingPaths.append([pathName])

                        # # I want to add this block in but my server does not have the RAM required to run it.
                        # # There may be another issue at play but that is my first guess.
                        # if patient_num in IPH_patients:
                        #     imageA = image
                        #     for x in range(0, 2):
                        #         lock.release()
                        #         imageA[:, :, :-1] = imageReduc(imageA[:, :, :-1])
                        #         pathName_iph = pathName + "iph_{}".format(x)
                        #         # print(pathName_iph)
                        #         lock.acquire()
                        #         trainingData.append([imageA])
                        #         trainingPaths.append([pathName_iph])

                        # if iLabel_num > 0.05:
                        #     imageA = image
                        #     for y in range(0, math.floor(iLabel_num / .03)):
                        #         lock.release()
                        #         imageA = dataAug(imageA)
                        #         lock.acquire()
                        #         pathNameB = pathName + "_{}".format(y)
                        #         # print(pathNameB)
                        #         trainingData.append([imageA])
                        #         trainingPaths.append([pathNameB])
                        # for z in range(0, 2):
                        #     lock.release()
                        #     image = dataAug(image)
                        #     pathName_aug = pathName + "aug_{}".format(z)
                        #     lock.acquire()
                        #     trainingData.append([image])
                        #     trainingPaths.append([pathName_aug])
                    lock.release()
                # for testing purposes only
        timeEnd[count] = time.time()
        print("Time for count {} = {} seconds".format(count, timeEnd[count] - timeStart[count]))
        # print("in fileloop - {}".format(trainingData))
        return

    # make sure that data gets looked at even
    pathlist = os.listdir(rawDataPath)
    pathlist = np.sort(pathlist)
    pathlist = shuffle(pathlist, random_state=20)
    t = 0
    pLength = len(pathlist)

    while count < pLength:
        processes = []
        while t < 10 and count < pLength:
            fpath = os.path.join(dataPaths, pathlist[count])
            patient_num = fpath[-3:]
            patient_num = int(patient_num)
            timeStart[count] = time.time()
            if patient_num not in bad_patients:
                p = multiprocessing.Process(target=fileLoop, args=(fpath, patient_num, iteration, 1, objective))
                p.start()
                processes.append(p)
            count += 1
            t += 1
        t = 0
        for process in processes:
            process.join()
            # if t == pLength:
            #     break

    # convert to numpy arrays, because the data is 4D
    trainingData = np.array(trainingData)
    testingData = np.array(testingData)
    # validationData = np.array(validationData)
    trainingPaths = np.array(trainingPaths)
    testingPaths = np.array(testingPaths)
    # validationPaths = np.array(validationPaths)

    random_seed = np.random.randint(0, 1e5)
    trainingData = shuffle(trainingData, random_state=random_seed)
    testingData = shuffle(testingData, random_state=random_seed // 17)
    trainingPaths = shuffle(trainingPaths, random_state=random_seed)
    testingPaths = shuffle(testingPaths, random_state=random_seed // 17)

    # let us see what we got
    # print(testingPaths)
    print("training {}".format(trainingData.shape))
    print("testing {}".format(testingData.shape))
    # print("validation {}".format(validationData.shape))

    print("saved in : {}".format(savePath))
    
    # save the data used to identify brain/no brain
    if (objective == 0):
        dataFolder = os.path.join(savePath, "brainMask")
    else:
        # save the data used to identify blood
        dataFolder = os.path.join(savePath, "blood")
    
    # save the data
    np.save(os.path.join(dataFolder, "TrainingData.npy"), trainingData)
    np.save(os.path.join(dataFolder, "TestingData.npy"), testingData)
    np.save(os.path.join(dataFolder, "TrainingPaths.npy"), trainingPaths)
    np.save(os.path.join(dataFolder, "TestingPaths.npy"), testingPaths)


if __name__ == '__main__':
    # path to raw data
    rawDataPath = config.RAW_DATA_PATH
    #path to saved processed data
    savePath = config.PROCESSED_NUMPY_PATH
    output2DImages(4, 1, rawDataPath, savePath)
    
    # save data for displaying the ultrasound cone
    axisPath = os.path.join(config.PROCESSED_NUMPY_PATH, "axis")
    if not os.path.isdir(axisPath):
        rand_input_file = random.choice(os.listdir(os.path.join(config.RAW_DATA_PATH, "DoD001")))
        rand_input_file = os.path.join(config.RAW_DATA_PATH, rand_input_file)
        FetchPolarAxis(rand_input_file, axisPath)


# FetchTimeData('/data/TBI/Datasets/PolarData/DoD001/DoD001_Ter001_RC1_Harmonics_Polar.mat')
