import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime
import os
import multiprocessing

patientNum = "024"
scanNum = "006"
scanType = "Z"
savePath = "/data/TBI/Datasets/Pictures/Evaluator/"
image_num = 0
countx = 0
county = 0
xAxisPath = '/data/TBI/Datasets/NPFiles/xAxis.npy'
yAxisPath = '/data/TBI/Datasets/NPFiles/yAxis.npy'
xAxis = np.load(xAxisPath)
yAxis = np.load(yAxisPath)
xAxis = xAxis.astype(int)
yAxis = yAxis.astype(int)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dispDict = {
    "prob": True,
    "true": True,
    "mask": False,
    "diff": False,
    "confusion": False,
    "probMap": False,
    "bMode": False
}


def findImage(test_set):
    global image_num
    st = test_set.shape
    for i in range(0, st[0]):
        test_name = test_set[i]
        # print("path = {}".format(test_name))
        test_name = str(test_name)
        testNum = test_name[5:8]
        if patientNum == testNum:
            if scanNum in test_name or scanType in test_name:
                print("Found patient {}".format(patientNum))
                print("Found scan {}".format(scanNum))
                print("setting image_num to {}".format(i))
                image_num = i
                return i


# import an x_train, x_test, y_train, y_test into this model.
# Patients from the validation set
# Need to implement the 3 layers to 1 on the label.
def preProcess(input_data, xdim=160, ydim=192):
    tt_y1 = input_data[:, :, :, :, range(0, 3)]
    input_data = np.delete(input_data, range(0, 3), 4)
    tt_x = input_data[:, :, :, :, :]
    tt_x = tt_x[image_num, :, :, :, :]
    tt_y = np.where(tt_y1[image_num, :, :, :, 1] > 0, 1, 0)
    tt_y = tt_y + tt_y1[image_num, :, :, :, 2]
    tt_y = np.where(tt_y1[image_num, :, :, :, 2] == 1, tt_y + 1, tt_y)
    tt_x = np.array(tt_x)
    tt_y = np.array(tt_y)
    tt_y = tt_y.reshape([xdim, ydim])
    tt_x = tt_x.reshape([1, xdim, ydim, -1])
    return tt_x, tt_y


def preProcess1(input_data, xdim=160, ydim=192):
    tt_y = input_data[image_num, :, :, :, 0]
    # print(tt_y.shape)
    input_data = np.delete(input_data, 0, 4)
    tt_x = input_data[:, :, :, :, :]
    tt_x = tt_x[image_num, :, :, :, :]
    tt_x = np.array(tt_x)
    tt_y = np.array(tt_y)
    tt_y = tt_y.reshape([xdim, ydim])
    tt_x = tt_x.reshape([1, xdim, ydim, -1])
    return tt_x, tt_y


def my_loss_cat(y_true, y_pred):
    class_factor = [1.1603, 0.50832, 5.8513]
    CE = 0.0
    scale_factor = 1 / tf.reduce_sum(y_true[:, :, :, :])
    for c in range(0, 3):
        # print(y_pred[:, :, :, c])
        CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            tf.math.log(y_pred[:, :, :, c]), tf.float32))) * scale_factor * class_factor[c]
    return CE * -1 * 3


def Polar_Model(sAll=False):
    global image_num
    dataPath_m = '/data/TBI/Datasets/NPFiles/IPH/TestingData.npy'
    pathNames_path = '/data/TBI/Datasets/NPFiles/IPH/TestingPaths.npy'
    data_m = np.load(dataPath_m)
    pathNames = np.load(pathNames_path)
    if not sAll:
        print(pathNames)
        index = findImage(pathNames)
        # print(pathNames)
        [testX, testY] = preProcess1(data_m, xdim=256, ydim=64)
        PolarProcess(testX, testY, name=pathNames[index])
    else:
        processes = []
        pathLen = len(pathNames)
        tmpcnt = 0
        while tmpcnt < pathLen:
            for i in range(tmpcnt, tmpcnt + 16):
                if i >= pathLen:
                    break
                image_num = i
                [testX, testY] = preProcess1(data_m, xdim=256, ydim=64)
                p = multiprocessing.Process(target=PolarProcess, args=(testX, testY, pathNames[i]))
                p.start()
                processes.append(p)
                print("Process Create")
            for process in processes:
                # print("Process Finished")
                process.join()
            tmpcnt = tmpcnt + 16
            print(tmpcnt)
        print(tmpcnt)
        # for i in range(0, len(pathNames)):
        #     image_num = i
        #     [testX, testY] = preProcess1(data_m, xdim=256, ydim=64)
        #     PolarProcess(testX, testY, name=pathNames[i])


def PolarProcess(testX, testY, name):
    # SegNet = tf.keras.models.load_model("tbi_ResNeSt_0")
    # DispInput(testX)
    SegNet = tf.keras.models.load_model("/data/TBI/Datasets/Models/IPH_Mobile_0", custom_objects={'my_loss_cat': my_loss_cat})
    prob = SegNet.predict(testX)
    probOut = prob[:, :, :, -1]
    prob = np.array(prob)
    prob = prob.reshape([256, 64, -1])
    bMode = testX[:, :, :, -1]

    probOut = np.array(probOut)
    probOut = probOut.reshape(256, 64)
    bMode = np.array(bMode)
    bMode = bMode.reshape(256, 64)

    probO = np.ones([256, 64])
    probO -= prob[:, :, 0]
    probO -= prob[:, :, 1] * .5
    probO += prob[:, :, 2]

    dispDict["probMap"] = True
    dispDict["bMode"] = True
    Display(probO, testY, name=name, numDim=3, bMode=bMode, probmap=probOut)
    return


def DispInput(x):
    xshape = x.shape
    for i in range(0, xshape[3]):
        tempx = x[:, :, :, i]
        plt.grid(False)
        plt.imshow(tempx, cmap='winter')
        path = savePath + datetime.datetime.now().strftime("%m%d-%H") + "/" + '2layer ' + str(i) + ".png"
        if not os.path.isdir(savePath + datetime.datetime.now().strftime("%m%d-%H")):
            os.mkdir(savePath + datetime.datetime.now().strftime("%m%d-%H"))
        if not os.path.isfile(path):
            plt.savefig(path)
        # plt.show()
        plt.close()


def Display(prob=None, true=None, mask=None, confusion=None, name="True", numDim=3, bMode=None, probmap=None):
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=.25, wspace=.3, bottom=.1)
    fig.set_size_inches(10, 6)
    name = str(name) + datetime.datetime.now().strftime("%f")
    global countx
    global county
    colormap = 'magma'
    if dispDict["prob"]:
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, prob, cmap=colormap)
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text('Prediction')
        checkCount(name)
    dispDict["prob"] = True

    if dispDict["true"]:
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, true, cmap=colormap, vmin=0, vmax=2)
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text(name)
        checkCount(name)
    dispDict["true"] = True

    if dispDict["mask"]:
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, mask, cmap=colormap, shading='auto', edgecolors='none')
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text('Brain_Mask')
        checkCount(name)
        dispDict["mask"] = False

    if dispDict["diff"]:
        diff = np.where(prob != true, 1, 0)
        diff = np.where(((true == numDim) & (prob != numDim)), numDim-1, diff)
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, diff, cmap=colormap, shading='auto', edgecolors='none')
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text('Difference')
        checkCount(name)
        dispDict["diff"] = False

    if dispDict["confusion"]:
        ax[county, countx].imshow(confusion, interpolation='nearest', cmap='ocean')
        ax[county, countx].set_ylabel('True label')
        ax[county, countx].set_xlabel('Predicted label')
        ax[county, countx].title.set_text('Confusion Matrix')
        checkCount(name)
        dispDict["confusion"] = False

    if dispDict["probMap"]:
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, probmap, cmap=colormap, vmin=0, vmax=1)
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text('Probability Bleed')
        checkCount(name)
        dispDict["probMap"] = False

    if dispDict["bMode"]:
        _, bin_edges = np.histogram(bMode, bins=25)
        # print(bin_edges)
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, bMode, cmap='binary', vmin=bin_edges[2], vmax=bin_edges[-2])
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text('bMode')
        checkCount(name)
        dispDict["bMode"] = False

    path = savePath + datetime.datetime.now().strftime("%m%d-%H") + "/" + str(name) + ".png"
    if not os.path.isdir(savePath + datetime.datetime.now().strftime("%m%d-%H")):
        os.mkdir(savePath + datetime.datetime.now().strftime("%m%d-%H"))
    if not os.path.isfile(path):
        print(name)
        plt.savefig(path)
    # plt.show()
    plt.close()
    countx = 0
    county = 0


def checkCount(name="name"):
    global countx
    global county
    countx += 1
    if countx == 2:
        county += 1
        if county == 2:
            path = savePath + datetime.datetime.now().strftime("%m%d-%H") + "/" + str(name) + ".png"
            if not os.path.isdir(savePath + datetime.datetime.now().strftime("%m%d-%H")):
                os.mkdir(savePath + datetime.datetime.now().strftime("%m%d-%H"))
            plt.savefig(path)
            # plt.show()
            county = 0
        countx = 0


Polar_Model(sAll=True)
# CT_Derived_Model()
# segNet_Transfer_Model()
# Split_Model()