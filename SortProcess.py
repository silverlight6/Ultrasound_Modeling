import numpy as np
import tensorflow as tf
import os


def my_loss_cat(y_true, y_pred):
    class_factor = [1.1603, 0.50832, 5.8513]
    CE = 0.0
    scale_factor = 1 / tf.reduce_sum(y_true[:, :, :, :])
    for c in range(0, 3):
        CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            tf.math.log(y_pred[:, :, :, c]), tf.float32))) * scale_factor * class_factor[c]
    return CE * -1 * 3


def sort_data(data, paths):
    finalData = []
    pathFinal = []
    pathLen = len(paths)
    startIndex = 0
    t_index = 0
    batch = 512
    temp_data = np.zeros([1, 256, 64, 17])
    while t_index < pathLen:
        if t_index + batch > pathLen:
            batch = pathLen - t_index
        batch_data = data[t_index:t_index + batch]
        batch_data = batch_data.reshape([batch, 256, 64, 18])
        batch_data_y = batch_data[:, :, :, 0]
        batch_data = np.delete(batch_data, [0, 17], 3)
        batch_data_y = batch_data_y.reshape([batch, 256, 64, 1])
        print("t_index num {}".format(t_index))
        SegNet = tf.keras.models.load_model("/DATA/TBI/Datasets/Models/ResNeSt_D1",
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
        currentPath = str(paths[startIndex])
        finalData.append(data[startIndex, :, :, :])
        pathFinal.append(str(paths[startIndex]))
        paths[startIndex] = " "
        tmpIndex = 0
        while index < pathLen and tmpIndex < 9:
            if str(paths[index]) == currentPath:
                finalData.append(data[index, :, :, :])
                # pathFinal.append(str(currentPath))
                paths[index] = " "
                tmpIndex += 1
            index += 1
            while index < pathLen and paths[index] == " ":
                index += 1
        startIndex += 1
        # print(tmpIndex)
        while tmpIndex < 9:
            finalData.append(np.zeros([256, 64, 4]))
            # pathFinal.append(str(currentPath))
            tmpIndex += 1
        # print(tmpIndex)
        while startIndex < pathLen - 1 and paths[startIndex] == " ":
            startIndex += 1

    finalData = np.array(finalData)
    pathFinal = np.array(pathFinal)
    finalData = finalData.reshape([-1, 10, 256, 64, 4])
    print("Final shape")
    print(finalData.shape)
    print(pathFinal.shape)
    
    return finalData, pathFinal


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #
    # train_data = '/DATA/TBI/Datasets/NPFiles/DispBal/TrainingData.npy'
    # train_path = '/DATA/TBI/Datasets/NPFiles/DispBal/TrainingPaths.npy'
    # pathNames = np.load(train_path)
    # data = np.load(train_data)

    # finalData, pathFinal = sort_data(data, pathNames)
    savePath = "/DATA/TBI/Datasets/NPFiles/DispBal/"
    # np.save(savePath + "TrainingData2.npy", finalData)
    # np.save(savePath + "TrainingPaths2.npy", pathFinal)

    test_data = '/DATA/TBI/Datasets/NPFiles/DispBal/TestingData.npy'
    test_path = '/DATA/TBI/Datasets/NPFiles/DispBal/TestingPaths.npy'
    pathNames = np.load(test_path)
    data = np.load(test_data)

    finalData, pathFinal = sort_data(data, pathNames)
    np.save(savePath + "TestingData2.npy", finalData)
    np.save(savePath + "TestingPaths2.npy", pathFinal)

    validation_data = '/DATA/TBI/Datasets/NPFiles/DispBal/ValidationData.npy'
    validation_path = '/DATA/TBI/Datasets/NPFiles/DispBal/ValidationPaths.npy'
    pathNames = np.load(validation_path)
    data = np.load(validation_data)

    finalData, pathFinal = sort_data(data, pathNames)
    np.save(savePath + "ValidationData2.npy", finalData)
    np.save(savePath + "ValidationPaths2.npy", pathFinal)
    return 0


if __name__ == '__main__':

    main()
