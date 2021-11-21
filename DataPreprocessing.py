import numpy as np
import config
import os                           # reading files
import random
import cv2                          # resize
import math
import cmath
import multiprocessing
import random
import time
import matplotlib
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from scipy.io import loadmat

IPH_patients = [8, 9, 10, 12, 22, 47, 53, 62, 66, 67, 69, 74, 75, 78, 85, 89, 93,
                    101, 105, 107, 110, 112, 113, 120, 121, 126, 129, 130, 133]

bad_patients = [27, 28, 35, 36, 38, 49, 69, 90]

# Extract axis information to produce cone-shape images
def FetchAxis(datapath, axisPath):
    data = loadmat(datapath)

    xaxis = np.array(list(data['xAxis']))
    yaxis = np.array(list(data['zAxis']))

    xaxis = cv2.resize(xaxis, (80, 256), interpolation=cv2.INTER_AREA)
    yaxis = cv2.resize(yaxis, (80, 256), interpolation=cv2.INTER_AREA)

    xaxis += 100
    yaxis -= 4

    print("saved axis info in : {}".format(axisPath))
    np.save(os.path.join(axisPath, "xAxis.npy"), xaxis)
    np.save(os.path.join(axisPath, "yAxis.npy"), yaxis)


def extract_displace_data(rawData):
    """
    Extract the displacement data from a patient file.
    Ignore the last cardiac cycle.
    Take 10 points from the start of a cardiac cycle

    Args:
        rawData (raw from .mat): raw data loaded from .mat file

    Return:
        displace_data (numpy array): displacement data
    """
    displacement = np.array(list(rawData['displacement']))
    hrTimes = np.array(list(rawData['hrTimes']))
    hrShape = hrTimes.shape
    disShape = displacement.shape
    displace_data = np.zeros([disShape[0], disShape[1], 10, hrShape[1] - 1])
    for h in range(0, hrShape[1] - 1):
        start = int(math.ceil(30 * hrTimes[0, h]))
        displace_data[:, :, :, h] = displacement[:, :, start: start + 10]
    
    return np.array(displace_data.astype('float64'))


def normalize_displacement(displace_data):
    """
    Normalize displacement data

    Args:
        displace_data (numpy array): 10 points of displacement data in a cardiac cycle
    Return:
        displace_data (numpy array): one image = average of the 10 points
    """
    displace_data = displace_data - displace_data.mean(axis=0).mean(axis = 0)
    safe_max = np.abs(displace_data).max(axis=0).max(axis=0)
    safe_max[safe_max == 0] = 1
    displace_data = displace_data / safe_max 
    displace_data = cv2.resize(displace_data, (80, 256))

    return displace_data


def process_patients(path, objective, patient_num=None):
    """
    Process data of all patients

    Args:
        path (string): path to folder storing patient data
        objective (integer): 0-brain mask, 1-blood mask
        patient_num (int): not used currently
    Return:
        1. numpy array of displacement data, label, and bMode
        2. list of patient files
    """
    patients_data = []
    file_list = []
    for patient in os.listdir(path):
        dataPath = os.path.join(path, patient)
        data, fileNames = process_data(path=dataPath, objective=objective)
        patients_data.append(data)
        file_list.append(fileNames)
    
    return np.array(patients_data), np.array(file_list)
        

def process_data(path, objective):
    """
    Process the raw data for a patient
    """
    # list of input images
    data = []
    # list of file names
    fileNames = []
    # process all file in the directory
    for file in os.listdir(path):
        if ".mat" in file:
            print(file)               # to be removed
            filePath = os.path.join(path, file)
            fileName = file[0:17]
            rawData = loadmat(filePath)
            # get the labels
            normalMask = np.array(list(rawData['normalMask'])) 
            bloodMask = np.array(list(rawData['bloodMaskThick']))
            brainMask = np.array(list(rawData['brainMask']))
            bMode = np.array(list(rawData['bModeNorm']))

            if len(bloodMask) == 0:
                break
            
            # extract the displacement data
            displace_data = extract_displace_data(rawData)
            
            # resize the masks
            normalMask = cv2.resize(normalMask, (80, 256))
            bloodMask = cv2.resize(bloodMask, (80, 256))
            brainMask = cv2.resize(brainMask, (80, 256))
            bMode = np.log10(bMode)
            bMode = bMode.astype('float64')
            bMode = np.mean(bMode, axis=2)

            # create label
            if objective == 0:
                label = np.where(brainMask == 0, 0, 1)
            else:
                label = bloodMask + 1
                label = label.astype('float32')
                label = np.where(brainMask == 0, 0, label)

            label = label.reshape([256, 80, 1])
            n_cycles = displace_data.shape[-1]

            # normalize
            for k in range(0, n_cycles):
                displace_data0 = displace_data[:, :, :, k]
                displace_data0 = normalize_displacement(displace_data0)
                
                bMode0 = bMode[:, :, k]
                bMode0 = cv2.resize(bMode0, (80, 256))
                bMode0 = bMode0.reshape([256, 80, 1])

                if objective == 1:
                    # delete non-brain from input
                    for i in range(0, displace_data0.shape[-1]):
                        displace_data0[:, :, i] = np.where(brainMask == 0, 0.0, displace_data0[:, :, i])

                # concatenate into one structure
                sample = np.concatenate((label, displace_data0, bMode0), axis=2)
                
            data.append(sample)
            fileNames.append(fileName + '_cycle' + str(k))
            print("Data shape:", np.array(data).shape)
    return np.array(data), np.array(fileNames)


def split_data(data, nameList, savePath, test_size=0.2, random_state=42):
    """
    Split the data into train and test set and save to file
    
    Args:
        data (numpy array): the data to operate on
        pathList (numpy array): the list of file name correspond to the data
        test_size (float): the propotion of the test set
        random_state (int): maintain the consistency between the data and file names
    
    Return: None
    """
    train_data, test_data, train_list, test_list = train_test_split(
                                                        data, 
                                                        nameList,
                                                        test_size=test_size,
                                                        random_state=random_state
                                                    )
    np.save(os.path.join(savePath, "TrainingData.npy"), train_data)
    np.save(os.path.join(savePath, "TestingData.npy"), test_data)
    np.save(os.path.join(savePath, "TrainingPaths.npy"), train_list)
    np.save(os.path.join(savePath, "TestingPaths.npy"), test_list)


if __name__ == '__main__':
    # path to raw data
    rawDataPath = config.RAW_DATA_PATH
    #path to saved processed data
    savePath = config.PROCESSED_NUMPY_PATH

    # create directory if does not exist
    brain_folder = os.path.join(savePath, 'brain')
    bleed_folder =  os.path.join(savePath, 'bleed')
    if not(os.path.exists(brain_folder)):
        os.mkdir(brain_folder)
    if not(os.path.exists(bleed_folder)):
        os.mkdir(bleed_folder)

    # process data and label to identify brain / no brain
    data, fileList = process_patients(rawDataPath, objective=0)
    print("data shape {}".format(data.shape))
    print("file names {}".format(fileList.shape))
    split_data(data, fileList, brain_folder, test_size=0.5)

    # process data and labels to identify blood
    data, fileList = process_patients(rawDataPath, objective=1)
    print("data shape {}".format(data.shape))
    print("file names {}".format(fileList.shape))
    split_data(data, fileList, bleed_folder, test_size=0.5)

    # save data for displaying the ultrasound cone
    axisPath = os.path.join(savePath, "axis")
    if not os.path.isdir(axisPath):
        try:
            os.mkdir(axisPath)
        except OSError as error:
            print(error)
        rand_patient = random.choice(os.listdir(rawDataPath))
        rand_input_file = random.choice(os.listdir(os.path.join(rawDataPath, rand_patient)))
        rand_input_file = os.path.join(rawDataPath, rand_patient, rand_input_file)
        FetchAxis(rand_input_file, axisPath)

