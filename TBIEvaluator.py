import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib
from matplotlib import pyplot as plt
# from sklearn.metrics import confusion_matrix
import datetime
import os
import multiprocessing

patientNum = "099"
scanNum = "007"
scanType = "RO3"
savePath = "/home/silver/TBI/Pictures/"
image_num = 0
countx = 0
county = 0
xAxisPath = '/home/silver/TBI/NPFiles/xAxis.npy'
yAxisPath = '/home/silver/TBI/NPFiles/yAxis.npy'
xAxis = np.load(xAxisPath)
yAxis = np.load(yAxisPath)
xAxis = xAxis.astype(int)
yAxis = yAxis.astype(int)
matplotlib.use('Agg')


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
    y = input_data[image_num, :, :, :, 0]
    bMode = input_data[image_num, :, :, :, 11]
    x = input_data[image_num, :, :, :, 1:11]
    x = np.array(x)
    y = np.array(y)
    y = y.reshape([xdim, ydim])
    x = x.reshape([1, xdim, ydim, -1])
    return x, y, bMode


def preProcess2(input_data, xdim=256, ydim=80):
    y_tr = input_data[:, :, :, :, 0]
    train_data = np.delete(input_data, 0, 4)
    x_tr = np.array(train_data)
    y_tr = y_tr[:, 0, :, :]
    x_tr = x_tr[image_num, :, :, :, :]
    y_tr = y_tr[image_num, :, :]
    y_tr = y_tr.reshape([256, 80])
    x_tr = x_tr.astype(dtype=np.float64)
    return x_tr, y_tr


def CardiacPreProcess(input_data, xdim=256, ydim=80, paths=None):
    global image_num
    t_x = []
    t_p = []
    t_y = input_data[image_num, :, :, 0]
    t_x.append(input_data[image_num, :, :, 1:16])
    t_p.append(paths[image_num])
    path = paths[image_num]
    image_num += 1
    while path == paths[image_num]:
        t_x.append(input_data[image_num, :, :, 1:16])
        t_p.append(paths[image_num])
        image_num += 1
    t_x = np.array(t_x)
    t_y = np.array(t_y)
    t_p = np.array(t_p)
    t_y = t_y.reshape([xdim, ydim])
    t_x = t_x.reshape([-1, xdim, ydim, 3])
    t_p = t_p.reshape([-1])
    return t_x, t_y, t_p


def my_loss_cat(y_true, y_pred):
    class_factor = [1.1603, 0.50832, 5.8513]
    CE = 0.0
    scale_factor = 1 / tf.reduce_sum(y_true[:, :, :, :])
    for c in range(0, 3):
        # print(y_pred[:, :, :, c])
        CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            tf.math.log(y_pred[:, :, :, c]), tf.float32))) * scale_factor * class_factor[c]
    return CE * -1 * 3


def Cardiac_Model():
    global image_num
    global lock
    dataPath_m = '/home/silver/TBI/NPFiles/Disp/TestingData2.npy'
    pathNames_path = '/home/silver/TBI/NPFiles/Disp/TestingPaths2.npy'
    data_m = np.load(dataPath_m)
    pathNames = np.load(pathNames_path)
    processes = []
    # pathLen = len(pathNames) / 10
    pathLen = len(pathNames)
    tmpcnt = 0
    lock = multiprocessing.Lock()
    while image_num < pathLen:
        while image_num < tmpcnt + 16 and image_num < pathLen:
            if image_num >= pathLen:
                break
            lock.acquire()
            [testX, testY] = preProcess2(data_m, xdim=256, ydim=80)
            lock.release()
            p = multiprocessing.Process(target=PolarProcess, args=(testX, testY, pathNames[image_num]))
            p.start()
            processes.append(p)
            print("Process Create")
            image_num += 1
        for process in processes:
            process.join()
        tmpcnt = tmpcnt + 16
        print(tmpcnt)
    print(tmpcnt)


def Polar_Model(sAll=False):
    global image_num
    dataPath_m = '/home/silver/TBI/NPFiles/Disp/TestingData.npy'
    pathNames_path = '/home/silver/TBI/NPFiles/Disp/TestingPaths.npy'
    data_m = np.load(dataPath_m)
    pathNames = np.load(pathNames_path)
    if not sAll:
        print(pathNames)
        index = findImage(pathNames)
        # print(pathNames)
        [testX, testY, bMode] = preProcess1(data_m, xdim=256, ydim=80)
        PolarProcess(testX, testY, name=pathNames[index], bMode=bMode)
    else:
        pathLen = len(pathNames)
        tmpcnt = 0
        while tmpcnt < pathLen:
            processes = []
            for i in range(tmpcnt, tmpcnt + 16):
                if i >= pathLen:
                    break
                image_num = i
                [testX, testY, bMode] = preProcess1(data_m, xdim=256, ydim=80)
                p = multiprocessing.Process(target=PolarProcess, args=(testX, testY, pathNames[i], bMode))
                p.start()
                processes.append(p)
                print("Process Create")
            for process in processes:
                process.join()
            tmpcnt = tmpcnt + 16
            print(tmpcnt)
        print(tmpcnt)


def PolarProcess(testX, testY, name, bMode=None, paths=None):

    # DispInput(testX)
    SegNet = tf.keras.models.load_model("/home/silver/TBI/Models/ResNeSt_T0",
                                        custom_objects={'my_loss_cat': my_loss_cat,
                                                        'add_ons>AdamW': tfa.optimizers.AdamW})
    # SegNet = tf.keras.models.load_model("/TBI/Models/Transformer_1")
    # prob = SegNet({"imp0": tf.expand_dims(testX[0, :, :, :], 0), "imp1": tf.expand_dims(testX[1, :, :, :], 0),
    #                "imp2": tf.expand_dims(testX[2, :, :, :], 0), "imp3": tf.expand_dims(testX[3, :, :, :], 0),
    #                "imp4": tf.expand_dims(testX[4, :, :, :], 0), "imp5": tf.expand_dims(testX[5, :, :, :], 0),
    #                "imp6": tf.expand_dims(testX[6, :, :, :], 0), "imp7": tf.expand_dims(testX[7, :, :, :], 0),
    #                "imp8": tf.expand_dims(testX[8, :, :, :], 0), "imp9": tf.expand_dims(testX[9, :, :, :], 0)})
    # Next 6 lines are for using the model generated model.
    # mask, _ = SegNet(testX)
    # mask = np.array(mask)
    # mask = np.round(mask)
    # for i in range(0, 10):
    #     testX[:, :, :, i] = np.where(mask[0, :, :, 0] == 1, 0.0, testX[:, :, :, i])
    # SegNet = tf.keras.models.load_model("/home/silver/TBI/Models/ResNeSt_T1-9",
    #                                     custom_objects={'my_loss_cat': my_loss_cat,
    #                                                     'add_ons>AdamW': tfa.optimizers.AdamW})

    prob, _ = SegNet(testX)
    probOut = prob[:, :, :, -1]
    prob = np.array(prob)
    prob = prob.reshape([256, 80, -1])

    probOut = np.array(probOut)
    probOut = probOut.reshape(256, 80)
    bMode = np.array(bMode)
    bMode = bMode.reshape(256, 80)
    bMode = bMode * -1

    probO = np.ones([256, 80])
    probO -= prob[:, :, 0]
    probO -= prob[:, :, 1] * .5
    probO += prob[:, :, 2]

    dispDict["probMap"] = True
    dispDict["bMode"] = True
    # print(testY.shape)
    Display(probO, testY, name=name, numDim=2, bMode=bMode, probmap=probOut)
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
        ax[county, countx].pcolormesh(xAxis, yAxis, prob, shading='flat', cmap=colormap, vmin=0, vmax=2)
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text('Prediction')
        checkCount(name)
    dispDict["prob"] = True

    if dispDict["true"]:
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, true, shading='flat', cmap=colormap, vmin=0, vmax=2)
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text(name)
        checkCount(name)
    dispDict["true"] = True

    if dispDict["mask"]:
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, mask, shading='flat', cmap=colormap, edgecolors='none')
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text('Brain_Mask')
        checkCount(name)
        dispDict["mask"] = False

    if dispDict["diff"]:
        diff = np.where(prob != true, 1, 0)
        diff = np.where(((true == numDim) & (prob != numDim)), numDim-1, diff)
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, diff, shading='flat', cmap=colormap, edgecolors='none')
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
        ax[county, countx].pcolormesh(xAxis, yAxis, probmap, shading='flat', cmap=colormap, vmin=0, vmax=1)
        ax[county, countx].invert_yaxis()
        ax[county, countx].title.set_text('Probability Bleed')
        checkCount(name)
        dispDict["probMap"] = False

    if dispDict["bMode"]:
        _, bin_edges = np.histogram(bMode, bins=25)
        ax[county, countx].grid(False)
        ax[county, countx].pcolormesh(xAxis, yAxis, bMode, shading='flat', cmap='binary',
                                      vmin=bin_edges[2], vmax=bin_edges[-2])
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
            plt.close()
            county = 0
        countx = 0


# Cardiac_Model()
Polar_Model(sAll=True)
# CT_Derived_Model()
# segNet_Transfer_Model()
# Split_Model()
