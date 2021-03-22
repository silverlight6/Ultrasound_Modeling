import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pathNames_data = '/data/TBI/Datasets/NPFiles/Cardiac/TestingData.npy'
pathNames_path = '/data/TBI/Datasets/NPFiles/Cardiac/TestingPaths.npy'
pathNames = np.load(pathNames_path)
data = np.load(pathNames_data)
print("first 20 paths {}".format(pathNames[0:20]))
print(data.shape)
trainData = []
pathFinal = []
pathLen = len(pathNames)
startIndex = 0

t_index = 0
batch = 512
temp_data = np.zeros([1, 256, 64, 16])


def my_loss_cat(y_true, y_pred):
    class_factor = [1.1603, 0.50832, 5.8513]
    CE = 0.0
    scale_factor = 1 / tf.reduce_sum(y_true[:, :, :, :])
    for c in range(0, 3):
        # print(y_pred[:, :, :, c])
        CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            tf.math.log(y_pred[:, :, :, c]), tf.float32))) * scale_factor * class_factor[c]
    return CE * -1 * 3


while t_index < pathLen:
    if t_index + batch > pathLen:
        batch = pathLen - t_index
    batch_data = data[t_index:t_index + batch]
    batch_data = batch_data.reshape([batch, 256, 64, 16])
    batch_data_y = batch_data[:, :, :, 0]
    batch_data = np.delete(batch_data, 0, 3)
    batch_data_y = batch_data_y.reshape([batch, 256, 64, 1])
    print("t_index num {}".format(t_index))
    SegNet = tf.keras.models.load_model("/data/TBI/Datasets/Models/ResNeSt_C1",
                                        custom_objects={'my_loss_cat': my_loss_cat})
    batch_data = SegNet.predict(batch_data)
    batch_data = np.concatenate((batch_data_y, batch_data), axis=3)
    print(batch_data.shape)
    if t_index == 0:
        temp_data = batch_data
    else:
        temp_data = np.concatenate((temp_data, batch_data), axis=0)
    t_index += batch

print(temp_data.shape)
data = temp_data

while startIndex < pathLen - 1:
    index = startIndex + 1
    currentPath = str(pathNames[startIndex])
    trainData.append(data[startIndex, :, :, :])
    pathFinal.append(str(pathNames[startIndex]))
    pathNames[startIndex] = " "
    while index < pathLen:
        if str(pathNames[index]) == currentPath:
            trainData.append(data[index, :, :, :])
            pathFinal.append(str(currentPath))
            pathNames[index] = " "
        index += 1
        while index < pathLen and pathNames[index] == " ":
            index += 1
    startIndex += 1
    while startIndex < pathLen - 1 and pathNames[startIndex] == " ":
        startIndex += 1

trainData = np.array(trainData)
print(trainData.shape)
print(pathFinal[0:200])

savePath = "/data/TBI/Datasets/NPFiles/Cardiac/"
np.save(savePath + "TestingData2.npy", trainData)
np.save(savePath + "TestingPaths2.npy", pathFinal)

