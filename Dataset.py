import numpy as np
import tensorflow as tf


class Dataset(object):

    def __init__(self, train_path=None, val_path=None, batch_size = 16):

        print("\nInitializing Dataset...")

        self.OUTPUT_CHANNELS = 3
        self.shape = [256, 80, 10]
        self.BATCH_SIZE = batch_size

        self.train_data = tf.data.Dataset.from_tensor_slices(np.load(train_path))
        self.test_data = tf.data.Dataset.from_tensor_slices(np.load(val_path))
        self.train_data = self.train_data.map(self.preProcess, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_data = self.test_data.map(self.preProcess, num_parallel_calls=tf.data.AUTOTUNE)

        # self.train_data.shuffle(tf.data.AUTOTUNE)
        self.train_data.batch(self.BATCH_SIZE)

        # self.test_data.shuffle(tf.data.AUTOTUNE)
        self.test_data.batch(1)

    def preProcess(self, input_data):
        t_y = tf.gather(input_data, 0, axis=3)  # weeding out the labels
        t_x = tf.gather(input_data, list(range(1, 11)), axis=3)  # weeding out the x data
        t_y = tf.cast(t_y, dtype=tf.float32)  # choose int32 types for the data
        tf_shape = t_x.shape
        t_x = tf.reshape(t_x, [tf_shape[0], self.shape[0], self.shape[1], self.shape[2]])
        t_y = tf.reshape(t_y, [tf_shape[0], self.shape[0], self.shape[1], 1])
        return t_x, t_y  # return input and output

    def label2vec(self, label):
        label = label.numpy()
        # print("label shape in label to vec = {}".format(label.shape))
        class_2 = np.where(label >= 1.05, label - 1, 0)
        class_2 = np.where(class_2 > 1, 1, class_2)
        class_1 = np.expand_dims(np.where(label > 0.95, 1 - class_2, 0), axis=3)
        class_0 = np.expand_dims(np.where(label <= 0.95, 1, 0), axis=3)
        class_2 = np.expand_dims(class_2, axis=3)
        label = np.concatenate((class_0, class_1, class_2), axis=3)
        label = np.reshape(label, [-1, self.shape[0], self.shape[1], 3])
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        return label
