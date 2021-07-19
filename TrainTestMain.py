# # Code is based on the github --> https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling.py
# # There will be A LOT of changes to it but the general structure is very similar
# # Biggest change is moving everything from pytorch to tensorflow
# # Second biggest change is moving from using a prebuilt ResNet base to a untrained and hand coded ResNeSt base.
# # The third change is I don't do a downsampling when flattening for patches to the transformer like in the paper

import datetime
import tensorflow as tf
# import os

from Dataset import Dataset
from VisionTransformer import VisionTransformer
from DataAugs import dataAug

mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
# wDecay = tf.keras.regularizers.L2(l2=0.00001)
wDecay = None
# tf.compat.v1.enable_eager_execution()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Process(object):
    def __init__(self, batch_size=16):
        self.summary_writer = tf.summary.create_file_writer("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.precision = tf.keras.metrics.Precision(name='precision')
        self.recall = tf.keras.metrics.Recall(name='recall')
        self.pre_c2 = tf.keras.metrics.Precision(name='precision_c2')
        self.re_c2 = tf.keras.metrics.Recall(name='recall_c2')
        self.mio = tf.keras.metrics.MeanIoU(name='mean_iou', num_classes=3)
        self.tr_recall = tf.keras.metrics.Recall(name='tr_recall')
        self.tr_precision = tf.keras.metrics.Precision(name='tr_precision')
        self.tr_mio = tf.keras.metrics.MeanIoU(name='tr_mio', num_classes=3)
        self.batch_size = batch_size

    def training(self, neuralnet, dataset, epochs, batch_size):
        print("\nTraining to %d epochs (%d of minibatch size)" % (epochs, batch_size))

        global wDecay
        iteration = 0
        prev_loss = 0
        loss = 1e7
        class_score = 0
        # tf.keras.utils.plot_model(neuralnet.resModel, to_file='ResNeSt.png', show_shapes=True)

        for epoch in range(epochs):
            # only useful if using a diminishing learning rate
            if neuralnet.learning_rate < 1e-5:
                break
            # Get a batch of data
            for (x_tr, y_tr) in dataset.train_data.map(lambda x, y: tf.py_function(dataAug, [x, y],
                                                                                   [tf.float64, tf.float32]),
                                                       num_parallel_calls=tf.data.AUTOTUNE):
                y_tr = dataset.label2vec(y_tr)

                # Take a step from that batch
                loss, class_score = self.mirrored_step(neuralnet, x_tr, y_tr, True)

                iteration += 1
                class_score_M = tf.math.round(class_score)
                y_tr_M = tf.math.round(y_tr)
                self.tr_recall.update_state(y_tr_M, class_score_M)
                self.tr_precision.update_state(y_tr_M, class_score_M)
                self.tr_mio.update_state(y_tr_M, class_score_M)
                if iteration % 491 == 0:
                    with self.summary_writer.as_default():
                        tf.summary.image("Train" + str(iteration), tf.concat([class_score, y_tr], axis=2),
                                         max_outputs=5, step=iteration)
                print('.', end='')
                if (iteration + 1) % 100 == 0:
                    print()

            # dataset.train_data.shuffle()
            print()
            print("Epoch [%d / %d] (%d iteration)  Loss:%.5f, Recall:%.5f, Precision:%.5f, IoU:%.5f"
                  % (epoch, epochs, iteration, loss, self.tr_recall.result(),
                     self.tr_precision.result(), self.tr_mio.result()))
            self.tr_recall.reset_states()
            self.tr_precision.reset_states()
            self.tr_mio.reset_states()
            if prev_loss == loss:
                print("Model is throwing a fit")
                print(class_score)
            prev_loss = loss
            self.test(neuralnet, dataset, epoch)
            # test(neuralnet, dataset, epoch)
            schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
                [10000, 20000, 40000, 60000, 80000], [1e-0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3])
            # lr and wd can be a function or a tensor
            neuralnet.learning_rate = 1e-3 * schedule(iteration)
            neuralnet.weight_decay = lambda: 1e-4 * schedule(iteration)
            wDecay = lambda: 1e-4 * schedule(iteration)

    def test(self, neuralnet, dataset, epoch):
        print("\nTest...")
        # Much of this code is copied from the ResNest source I found and then translated to Keras
        # This simply calculates the metrics listed at the top using the logits and true
        # I do not know if this is 100% bug free since switching to probability labels
        total_loss = 0
        test_iter = 0
        for (x_te, y_te) in dataset.test_data:
            y_te = dataset.label2vec(y_te)
            loss, class_score = self.mirrored_step(neuralnet, x_te, y_te, False)
            # loss, class_score = neuralnet.step(x=x_te, y=y_te, train=False)
            loss = tf.reduce_sum(loss)
            class_score_M = tf.math.round(class_score)
            y_te_M = tf.math.round(y_te)
            self.precision.update_state(y_te_M, class_score_M)
            self.recall.update_state(y_te_M, class_score_M)
            self.pre_c2.update_state(y_te_M[:, :, -1], class_score_M[:, :, -1])
            self.re_c2.update_state(y_te_M[:, :, -1], class_score_M[:, :, -1])
            self.mio.update_state(y_te_M, class_score_M)
            total_loss += loss
            if test_iter % 23 == 0:
                with self.summary_writer.as_default():
                    tf.summary.image("Test" + str(test_iter), tf.concat([class_score, y_te], axis=2), step=test_iter)
            test_iter += 1

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
    def mirrored_step(self, neuralnet, x, y, train):
        # x = mirrored_strategy.experimental_distribute_dataset(x)
        # y = mirrored_strategy.experimental_distribute_dataset(y)
        data = tf.data.Dataset.from_tensors(x).batch(self.batch_size)
        mirror_data = mirrored_strategy.experimental_distribute_dataset(data)
        loss, class_score = mirrored_strategy.run(neuralnet.step, args=(mirror_data, train,))
        loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        class_score = mirrored_strategy.reduce(tf.distribute.ReduceOp.Sum, class_score, axis=None)
        return loss, class_score


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_data = '/home/silver/TBI/NPFiles/Disp/TestingData-2.npy'
    val_data = '/home/silver/TBI/NPFiles/Disp/TestingData-2.npy'
    batch_size = 16
    dataset = Dataset(train_data, val_data, batch_size)
    # config = tf.estimator.RunConfig(train_distribute=mirrored_strategy)
    # with mirrored_strategy.scope():
    neuralnet = VisionTransformer()
    # tf.keras.utils.model_to_dot(neuralnet.visionModel, to_file='TransUNet_dot.png', show_shapes=True)
    # print(neuralnet.visionModel.summary())
    print(len(neuralnet.visionModel.layers))
    process = Process(batch_size)
    process.training(neuralnet=neuralnet, dataset=dataset, epochs=51, batch_size=batch_size)
    neuralnet.visionModel.save('/home/silver/TBI/Models/ResNeSt_T1-2')


if __name__ == '__main__':
    main()
