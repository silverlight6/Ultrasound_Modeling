import numpy as np
import config
import os                           # reading files
import cv2                          # resize
import math
import multiprocessing
import time
import matplotlib
from sklearn.utils import shuffle   # shuffle seed

from matplotlib import pyplot as plt
from scipy.io import loadmat

matplotlib.use('Agg')


def output2DImages(iteration):
    # Make arrays for dividing data between training, testing, and validation.
    manager = multiprocessing.Manager()

    trainingData = manager.list()
    testingData = manager.list()
    trainingPaths = manager.list()
    testingPaths = manager.list()

    IPH_patients = [8, 9, 10, 12, 24, 47, 53, 62, 66, 67, 69, 74, 75, 78, 85, 89, 93,
                    101, 105, 107, 110, 112, 113, 120, 121, 126, 129, 130, 133]
    bad_patients = [1, 14, 22, 23, 27, 28, 32, 34, 35, 36, 37, 38, 39, 44, 49, 69, 71, 78, 82, 90, 98, 101, 121, 124,
                    128, 133, 136, 928]
    timeStart = np.zeros([100])
    timeEnd = np.zeros([100])
    # counting files
    count = 0

    def fileLoop(path, patient_num, iteration, mode):
        iteration = iteration % 10
        for file in os.listdir(path):
            if ".mat" in file:
                hPath = os.path.join(path, file)                  # put file name on path
                pathName = file[0:17]                             # get path to print
                Harmonics = loadmat(hPath)
                bloodMask = np.array(list(Harmonics['bloodMaskThick']))
                brainMask = np.array(list(Harmonics['brainMask']))
                bMode = np.array(list(Harmonics['bModeNorm']))
                # print("Keys are {}".format(Harmonics.keys()))
                if len(bloodMask) == 0:
                    break
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

                bMode = np.log10(bMode)

                # Smooth & Create the Label
                label = bloodMask + 1
                label = label.astype('float32')
                # print(label.shape)
                brainMask = cv2.resize(brainMask, (80, 256))
                label = cv2.resize(label, (80, 256))
                label = np.where(brainMask == 0, 0, label)
                # # This code is for finding the mask
                # label = np.where(brainMask == 0, 0, 1)
                label = label.reshape([256, 80, 1])
                cycles = real.shape[-1]
                real = real.astype('float64')
                imag = imag.astype('float64')
                bMode = bMode.astype('float64')
                bMode = np.mean(bMode, axis=2)

                for k in range(0, cycles):
                    realO = real[:, :, :, k]
                    imagO = imag[:, :, :, k]
                    bModeO = bMode[:, :, k]

                    realO = realO - realO.mean(axis=0).mean(axis=0)
                    safe_max = np.abs(realO).max(axis=0).max(axis=0)
                    safe_max[safe_max == 0] = 1
                    realO = realO / safe_max

                    imagO = imagO - imagO.mean(axis=0).mean(axis=0)
                    safe_max = np.abs(imagO).max(axis=0).max(axis=0)
                    safe_max[safe_max == 0] = 1
                    imagO = imagO / safe_max

                    realO = cv2.resize(realO, (80, 256))
                    imagO = cv2.resize(imagO, (80, 256))
                    bModeO = cv2.resize(bModeO, (80, 256))

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

                    lock.release()
                # for testing purposes only
        timeEnd[count] = time.time()
        print("Time for count {} = {} seconds".format(count, timeEnd[count] - timeStart[count]))
        return

    # data paths; it is just what it sounds like
    # watch out for polar versus non-polar
    dataPaths = config.RAW_DATA_PATH

    # make sure that data gets looked at even

    pathlist = os.listdir(dataPaths)
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
            if patient_num not in bad_patients and patient_num in IPH_patients:
                p = multiprocessing.Process(target=fileLoop, args=(fpath, patient_num, iteration, 1))
                p.start()
                processes.append(p)
            count += 1
            t += 1
        t = 0
        for process in processes:
            process.join()

    # convert to numpy arrays, because the data is 4D
    trainingData = np.array(trainingData)
    testingData = np.array(testingData)
    trainingPaths = np.array(trainingPaths)
    testingPaths = np.array(testingPaths)

    random_seed = np.random.randint(0, 1e5)
    trainingData = shuffle(trainingData, random_state=random_seed)
    testingData = shuffle(testingData, random_state=random_seed // 17)
    trainingPaths = shuffle(trainingPaths, random_state=random_seed)
    testingPaths = shuffle(testingPaths, random_state=random_seed // 17)

    # let us see what we got
    # print(testingPaths)
    print("training {}".format(trainingData.shape))
    print("testing {}".format(testingData.shape))

    savePath = os.path.join(config.PROCESSED_NUMPY_PATH, 'pizza_IPH')
    print("saved in : {}".format(savePath))
    np.save(os.path.join(savePath, "TrainingData.npy"), trainingData)
    np.save(os.path.join(savePath, "TestingData.npy"), testingData)
    np.save(os.path.join(savePath, "TrainingPaths.npy"), trainingPaths)
    np.save(os.path.join(savePath, "TestingPaths.npy"), testingPaths)


if __name__ == '__main__':
    output2DImages(0)
