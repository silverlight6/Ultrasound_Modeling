import tensorflow as tf
import datetime
import os
from Dataset_2 import Dataset
from VisionTransformer import VisionTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
wDecay = None


class Process(object):
    def __init__(self, batch_size=16, num_classes=3):
        self.summary_writer = tf.summary.create_file_writer(
            "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.batch_size = batch_size
        self.input_shape = [256, 80, 10]
        self.precision = tf.keras.metrics.Precision(name='precision')
        self.recall = tf.keras.metrics.Recall(name='recall')
        self.pre_c2 = tf.keras.metrics.Precision(name='precision_c2')
        self.re_c2 = tf.keras.metrics.Recall(name='recall_c2')
        self.mio = tf.keras.metrics.MeanIoU(name='mean_iou', num_classes=num_classes)
        self.tr_recall = tf.keras.metrics.Recall(name='tr_recall')
        self.tr_precision = tf.keras.metrics.Precision(name='tr_precision')
        self.tr_mio = tf.keras.metrics.MeanIoU(name='tr_mio', num_classes=num_classes)

        self.test_iter = 0

    def training(self, neuralnet, dataset, epochs):

        print("\nTraining to %d epochs (%d of minibatch size)" % (epochs, self.batch_size))

        global wDecay
        iteration = 0
        prev_loss = 0
        # tf.keras.utils.plot_model(neuralnet.resModel, to_file='ResNeSt.png', show_shapes=True)

        for epoch in range(epochs):
            while True:
                # Get the data for the next batch
                x_tr, y_tr, terminator = dataset.next_train(self.batch_size)  # y_tr does not used in this prj.
                # Take a step from that batch
                loss, class_score = neuralnet.train_step(x=x_tr, y=y_tr)
                # Loss is 256x80 so reduce to 1 number
                loss = tf.reduce_sum(loss)
                iteration += 1
                class_score_M = tf.math.round(class_score)
                y_tr_M = tf.math.round(y_tr)
                self.tr_recall.update_state(y_tr_M, class_score_M)
                self.tr_precision.update_state(y_tr_M, class_score_M)
                self.tr_mio.update_state(y_tr_M, class_score_M)
                if iteration % 491 == 0:
                    with self.summary_writer.as_default():
                        # tf.summary.image("Train Logits" + str(iteration), class_score, max_outputs=5, step=iteration)
                        # tf.summary.image("Train Target" + str(iteration), y_tr, max_outputs=5, step=iteration)
                        tf.summary.image("Train" + str(iteration),
                                         tf.concat([class_score, y_tr], axis=2),
                                         max_outputs=5, step=iteration)

                print('.', end='')
                if (iteration + 1) % 100 == 0:
                    print()
                if terminator:
                    break

                # neuralnet.save_params()
            print()
            print("Epoch [%d / %d] (%d iteration)  Loss:%.5f, Recall:%.5f, Precision:%.5f, IoU:%.5f"
                  % (epoch, epochs, iteration, loss, self.tr_recall.result(), self.tr_precision.result(),
                     self.tr_mio.result()))
            self.tr_recall.reset_states()
            self.tr_precision.reset_states()
            self.tr_mio.reset_states()
            if prev_loss == loss:
                print("Model is throwing a fit")
                print(class_score)
            prev_loss = loss
            if epoch % 5 == 0:
                self.test(neuralnet, dataset, epoch)
            # test(neuralnet, dataset, epoch)
            schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
                [2000, 4000, 8000, 10000, 15000], [1e-0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3])
            # lr and wd can be a function or a tensor
            neuralnet.learning_rate = 1e-2 * schedule(iteration)
            neuralnet.weight_decay = lambda: 1e-4 * schedule(iteration)
            wDecay = lambda: 1e-4 * schedule(iteration)

    def test(self, neuralnet, dataset, epoch):
        print("\nTest...")
        # Much of this code is copied from the ResNest source I found and then translated to Keras
        # This simply calculates the metrics listed at the top using the logits and true
        # I do not know if this is 100% bug free since switching to probability labels
        total_loss = 0
        while True:
            x_te, y_te, terminator = dataset.next_test(16)  # y_te does not used in this prj.
            loss, class_score = neuralnet.step(x=x_te, y=y_te)
            loss = tf.reduce_sum(loss)
            class_score_M = tf.math.round(class_score)
            y_te_M = tf.math.round(y_te)
            self.precision.update_state(y_te_M, class_score_M)
            self.recall.update_state(y_te_M, class_score_M)
            self.pre_c2.update_state(y_te_M[:, :, -1], class_score_M[:, :, -1])
            self.re_c2.update_state(y_te_M[:, :, -1], class_score_M[:, :, -1])
            self.mio.update_state(y_te_M, class_score_M)
            total_loss += loss
            if self.test_iter % 23 == 0:
                with self.summary_writer.as_default():
                    # tf.summary.image("Test Logits" + str(test_iter), class_score, step=test_iter)
                    # tf.summary.image("Test Target" + str(test_iter), y_te, step=test_iter)
                    tf.summary.image("Test" + str(self.test_iter), tf.concat([class_score, y_te], axis=2),
                                     step=self.test_iter)
            if terminator:
                break
            self.test_iter += 1

        # This half of the code prints the metrics to the screen and saves them to a log file.
        # Later, you can open them up on tensorboard to see the progress.
        # total_loss /= dataset.num_te
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
        # print("learning rate = {}".format(neuralnet.learning_rate))

        self.mio.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.pre_c2.reset_states()
        self.re_c2.reset_states()
        return f1


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_data = '/home/silver/TBI/NPFiles/Disp/TrainingData.npy'
    val_data = '/home/silver/TBI/NPFiles/Disp/TestingData.npy'
    num_class = 3
    dataset = Dataset(train_data, val_data, num_class)
    # config = tf.estimator.RunConfig(train_distribute=mirrored_strategy)
    # with mirrored_strategy.scope():
    batch_size = 32
    neuralnet = VisionTransformer(batch_size, num_classes=num_class)

    # tf.keras.utils.model_to_dot(neuralnet.visionModel, to_file='TransUNet_dot.png', show_shapes=True)

    # print(neuralnet.visionModel.summary())
    print(len(neuralnet.visionModel.layers))
    process = Process(batch_size=batch_size, num_classes=num_class)
    process.training(neuralnet=neuralnet, dataset=dataset, epochs=51)
    neuralnet.visionModel.save('/home/silver/TBI/Models/ResNeSt_T0')


if __name__ == '__main__':
    main()
