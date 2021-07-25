# # Code is based on the github --> https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling.py
# # There will be A LOT of changes to it but the general structure is very similar
# # Biggest change is moving everything from pytorch to tensorflow
# # Second biggest change is moving from using a prebuilt ResNet base to a untrained and hand coded ResNeSt base.
# # The third change is I don't do a downsampling when flattening for patches to the transformer like in the paper

import os
import datetime
import tensorflow as tf

from Dataset import Dataset
from VisionTransformer import VisionTransformer
from DataAugs import dataAug

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mirrored_strategy = tf.distribute.MirroredStrategy()

# wDecay = tf.keras.regularizers.L2(l2=0.00001)
wDecay = None
# tf.compat.v1.enable_eager_execution()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Process(object):
    def __init__(self, batch_size=16):
        self.summary_writer = tf.summary.create_file_writer(
            "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.batch_size = batch_size
        self.input_shape = [256, 80, 10]
        self.precision = 0
        self.recall = 0
        self.pre_c2 = 0
        self.re_c2 = 0
        self.mio = 0
        self.tr_recall = 0
        self.tr_precision = 0
        self.tr_mio = 0
        self.train_iter = tf.constant(0)
        self.test_iter = tf.constant(0)

    def training(self, neuralnet, dataset, epochs, batch_size):
        print("\nTraining to %d epochs (%d of minibatch size)" % (epochs, batch_size))
        global wDecay
        prev_loss = 0
        # tf.keras.utils.plot_model(neuralnet.resModel, to_file='ResNeSt.png', show_shapes=True)

        for epoch in range(epochs):
            # only useful if using a diminishing learning rate
            if neuralnet.learning_rate < 1e-5:
                break
            # Get a batch of data
            datasetTemp = dataset.train_data.map(lambda x, y: tf.numpy_function(func=dataAug, inp=(x, y),
                                                                                Tout=(tf.float64, tf.float32)),
                                                 num_parallel_calls=tf.data.AUTOTUNE)
            datasetTemp = datasetTemp.map(lambda x, y: tf.numpy_function(func=Dataset.label2vec, inp=(x, y),
                                                                         Tout=(tf.float64, tf.float32)),
                                          num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            # print("or is it in the parallel file?")
            # Take a step from that batch
            loss = self.mirrored_train_step(neuralnet, datasetTemp)
            # dataset.train_data.shuffle()
            print()
            print("Epoch [%d / %d] (%d iteration)  Loss:%.5f, Recall:%.5f, Precision:%.5f, IoU:%.5f"
                  % (epoch, epochs, self.train_iter, loss, self.tr_recall.result(),
                     self.tr_precision.result(), self.tr_mio.result()))
            self.tr_recall.reset_states()
            self.tr_precision.reset_states()
            self.tr_mio.reset_states()
            if prev_loss == loss:
                print("Model is throwing a fit")
            prev_loss = loss
            self.test(neuralnet, dataset.test_data, epoch)
            # test(neuralnet, dataset, epoch)
            schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
                [10000, 20000, 40000, 60000, 80000], [1e-0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3])
            # lr and wd can be a function or a tensor
            neuralnet.learning_rate = 1e-3 * schedule(self.train_iter)
            neuralnet.weight_decay = lambda: 1e-4 * schedule(self.train_iter)
            wDecay = lambda: 1e-4 * schedule(self.train_iter)

    def test(self, neuralnet, dataset, epoch):
        print("\nTest...")
        # Much of this code is copied from the ResNest source I found and then translated to Keras
        # This simply calculates the metrics listed at the top using the logits and true
        # I do not know if this is 100% bug free since switching to probability labels
        total_loss = self.mirrored_test_step(neuralnet, dataset)
        # This half of the code prints the metrics to the screen and saves them to a log file.
        # Later, you can open them up on tensorboard to see the progress.
        # total_loss /= len(list(dataset.test_data))
        f1 = 2 * (self.precision.result() * self.recall.result()) / (self.precision.result() + self.recall.result())
        f1_2 = 2 * (self.pre_c2.result() * self.re_c2.result()) / (self.pre_c2.result() + self.re_c2.result())

        with self.summary_writer.as_default():
            tf.summary.scalar("loss", total_loss, step=epoch)
            tf.summary.scalar("mean_IoU", self.mio.result(), step=epoch)
            tf.summary.scalar("val_f1", f1, step=epoch)
            tf.summary.scalar("val_precision", self.precision.result(), step=epoch)
            tf.summary.scalar("recall_recall", self.recall.result(), step=epoch)
            tf.summary.scalar("c2_f1", f1_2, step=epoch)
            tf.summary.scalar("c2_precision", self.pre_c2.result(), step=epoch)
            tf.summary.scalar("c2_recall", self.re_c2.result(), step=epoch)
            tf.summary.scalar("loss", total_loss, step=epoch)

        print("loss = {}".format(total_loss))
        print("IoU = {}".format(self.mio.result()))
        print("f1 = {}".format(f1))
        print("precision = {}".format(self.precision.result()))
        print("recall = {}".format(self.recall.result()))

        self.mio.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.pre_c2.reset_states()
        self.re_c2.reset_states()
        return f1

    @tf.function
    def mirrored_train_step(self, neuralnet, dataset):
        loss = 1e-8
        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # dataset = dataset.with_options(options)
        local_recall = []
        local_precision = []
        local_mio = []
        dataset = dataset.map(self.reshape_map, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        mirror_data = mirrored_strategy.experimental_distribute_dataset(dataset)
        for x, y in mirror_data:
            loss, class_score, recall, precision, mio = mirrored_strategy.run(neuralnet.train_step, args=(x, y,))
            loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            local_recall = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, recall, axis=None)
            local_precision = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, precision, axis=None)
            local_mio = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, mio, axis=None)
            self.train_iter += 1
            if self.train_iter % 491 == 0:
                with self.summary_writer.as_default():
                    tf.summary.image("Train" + str(self.train_iter), tf.concat([class_score, y], axis=2),
                                     max_outputs=5, step=tf.cast(self.train_iter, tf.int64))
            print('.', end='')
            if (self.train_iter + 1) % 100 == 0:
                print()
        self.recall = tf.reduce_mean(local_recall)
        self.precision = tf.reduce_mean(local_precision)
        self.mio = tf.reduce_mean(local_mio)
        return loss

    @tf.function
    def mirrored_test_step(self, neuralnet, dataset):
        total_loss = 0
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = dataset.map(self.reshape_map)
        dataset = dataset.batch(self.batch_size)
        mirror_data = mirrored_strategy.experimental_distribute_dataset(dataset)
        for x, y in mirror_data:
            loss, class_score = mirrored_strategy.run(neuralnet.step, args=(x, y))
            loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            class_score = mirrored_strategy.gather(class_score, axis=0)
            loss = tf.reduce_sum(loss)
            class_score_M = tf.math.round(class_score)
            y = mirrored_strategy.gather(y, axis=0)
            y_te_M = tf.math.round(y)
            self.precision.update_state(y_te_M, class_score_M)
            self.recall.update_state(y_te_M, class_score_M)
            self.pre_c2.update_state(y_te_M[:, :, -1], class_score_M[:, :, -1])
            self.re_c2.update_state(y_te_M[:, :, -1], class_score_M[:, :, -1])
            self.mio.update_state(y_te_M, class_score_M)
            total_loss += loss
            if self.test_iter % 23 == 0:
                with self.summary_writer.as_default():
                    tf.summary.image("Test" + str(self.test_iter), tf.concat([class_score, y], axis=2),
                                     step=tf.cast(self.test_iter, tf.int64))
            self.test_iter += 1
        return total_loss

    def reshape_map(self, x, y):
        # x = tf.squeeze(x)
        # y = tf.squeeze(y)
        x.set_shape(tf.TensorShape([None, None, None]))
        y.set_shape(tf.TensorShape([None, None, None]))
        x = tf.reshape(x, [self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        y = tf.reshape(y, [self.input_shape[0], self.input_shape[1], 3])
        return x, y

    # An example for the map command
    def local_dataAugs(self, x, y):
        x, y = tf.numpy_function(func=dataAug, inp=(x, y), Tout=(tf.float64, tf.float32))
        x.set_shape(tf.TensorShape([None, None, None]))
        y.set_shape(tf.TensorShape([None, None, None]))
        return x, y

    def local_label2vec(self, x, y):
        x, y = tf.numpy_function(func=Dataset.label2vec, inp=(x, y), Tout=(tf.float64, tf.float32))
        x.set_shape(tf.TensorShape([None, None, None]))
        y.set_shape(tf.TensorShape([None, None, None]))
        return x, y


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_data = '/home/silver/TBI/NPFiles/Disp/TestingData.npy'
    val_data = '/home/silver/TBI/NPFiles/Disp/TestingData.npy'
    batch_size = 64
    dataset = Dataset(train_data, val_data, batch_size)
    # config = tf.estimator.RunConfig(train_distribute=mirrored_strategy)
    # with mirrored_strategy.scope():
    with mirrored_strategy.scope():
        neuralnet = VisionTransformer(batch_size=batch_size)
    # tf.keras.utils.model_to_dot(neuralnet.visionModel, to_file='TransUNet_dot.png', show_shapes=True)
    # print(neuralnet.visionModel.summary())
    print(len(neuralnet.visionModel.layers))
    process = Process(batch_size)
    process.training(neuralnet=neuralnet, dataset=dataset, epochs=51, batch_size=batch_size)
    neuralnet.visionModel.save('/home/silver/TBI/Models/ResNeSt_T1')


if __name__ == '__main__':
    main()
