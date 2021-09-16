import tensorflow as tf
import numpy as np
from DataAugs import dataAug


def label2vec(label, num_classes):
    if num_classes == 3:
        class_2 = np.where(label >= 1.05, label - 1, 0)
        class_2 = np.where(class_2 > 1, 1, class_2)
        class_1 = np.expand_dims(np.where(label > 0.95, 1 - class_2, 0), axis=3)
        class_0 = np.expand_dims(np.where(label <= 0.95, 1, 0), axis=3)
        class_2 = np.expand_dims(class_2, axis=3)
        label = np.concatenate((class_0, class_1, class_2), axis=3)
    else:
        class_1 = label
        class_0 = 1 - label
        class_0 = np.expand_dims(class_0, axis=3)
        class_1 = np.expand_dims(class_1, axis=3)
        label = np.concatenate((class_0, class_1), axis=3)
    return label


class Dataset(object):

    def __init__(self, train_path=None, val_path=None, num_classes=3):

        print("\nInitializing Dataset...")
        train_data = np.load(train_path, allow_pickle=True)
        val_data = np.load(val_path, allow_pickle=True)

        # The first 0 is due to how the .append works in the playground file
        # The second 0 is because the label is in the first layer of the data.
        y_tr = train_data[:, 0, :, :, 0]
        y_te = val_data[:, 0, :, :, 0]
        train_data = np.delete(train_data, 0, 4)
        val_data = np.delete(val_data, 0, 4)
        x_tr = np.array(train_data)
        x_te = np.array(val_data)
        # The -1 here is because the last layer is the bMode and I am not using the bMode in the training data
        # This is simply my choice, feel free to change that but be aware that the number of input layers
        # Moves from 10 to 11 and that will affect some lines of code in the evaluator file.
        x_tr = x_tr[:, 0, :, :, :-1]
        x_te = x_te[:, 0, :, :, :-1]
        # This is float64 by default but needs to be float 32 for np.where function.
        y_tr = y_tr.astype(dtype=np.float32)
        y_te = y_te.astype(dtype=np.float32)

        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_te, self.y_te = x_te, y_te

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        self.y_tr = y_tr
        self.y_te = y_te

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        print("Number of data\nTraining: %d, Test: %d\n" % (self.num_tr, self.num_te))
        print("x_tr shape = {}".format(x_tr.shape))
        print("y_tr shape = {}".format(y_tr.shape))

        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        self.channel = x_sample.shape[2]

        self.min_val, self.max_val = x_sample.min(), x_sample.max()
        # self.num_class = int(np.floor(y_te.max()+1))
        self.num_classes = num_classes

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" % (self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" % (self.min_val, self.max_val))
        print("Class  %d" % self.num_classes)

    # This is converting the labels from 1d to 3d probability maps
    # 1.05 because resize can sometimes change values up and down by about .01 at any given pixel
    # if > 1 = 1 is because resize sometimes makes values at 101% up to 105%. Putting a cap to ensure
    # this behavior is stopped.
    # The classes don't need to add up to 100. It is simply a nice thing if they do.
    # You only really need each pixel of importance to have a contribution to the loss.

    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    # Get the next batch of training data
    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size
        x_tr, y_tr = np.copy(self.x_tr[start:end]), np.copy(self.y_tr[start:end])

        terminator = False
        if end >= self.num_tr:
            terminator = True
            self.idx_tr = 0
            # self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else:
            self.idx_tr = end

        if fix:
            self.idx_tr = start

        if x_tr.shape[0] != batch_size:
            x_tr, y_tr = np.copy(self.x_tr[-1-batch_size:-1]), np.copy(self.y_tr[-1-batch_size:-1])

        # data aug goes here.
        trShape = x_tr.shape
        for i in range(0, trShape[0]):
            x_tr[i, :, :, :], y_tr[i, :, :] = dataAug(x_tr[i, :, :, :], y_tr[i, :, :])

        y_tr = label2vec(y_tr, self.num_classes)
        y_tr = tf.convert_to_tensor(y_tr, dtype=tf.float32)
        return x_tr, y_tr, terminator

    # Get the next batch of test data. This is always 1 for batch size in this model up to this point
    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te + batch_size
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]

        terminator = False
        if end >= self.num_te:
            terminator = True
            self.idx_te = 0
        else:
            self.idx_te = end

        if x_te.shape[0] != batch_size:
            x_te, y_te = self.x_te[-1-batch_size:-1], self.y_te[-1-batch_size:-1]

        y_te = label2vec(y_te, self.num_classes)
        y_te = tf.convert_to_tensor(y_te, dtype=tf.float32)
        return x_te, y_te, terminator