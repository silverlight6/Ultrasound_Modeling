import tensorflow as tf
import numpy as np
import datetime
import os
#
# mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# tf.compat.v1.enable_eager_execution()
summary_writer = tf.summary.create_file_writer("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
precision = tf.keras.metrics.Precision(name='precision')
recall = tf.keras.metrics.Recall(name='recall')
pre_c2 = tf.keras.metrics.Precision(name='precision_c2')
re_c2 = tf.keras.metrics.Recall(name='recall_c2')


class ResNest:
    def __init__(self, height, width, channel, num_class,
                 ksize, radix=4, kpaths=4, learning_rate=1e-3, ckpt_dir='./Checkpoint'):

        print("\nInitializing Short-ResNeSt...")
        super(ResNest, self).__init__()
        self.height, self.width, self.channel, self.num_class = height, width, channel, num_class
        self.ksize, self.learning_rate = ksize, learning_rate
        self.radix, self.kpaths = radix, kpaths
        self.ckpt_dir = ckpt_dir

        self.resModel = self.model()

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        # self.optimizer = tf.optimizers.Adamax(self.learning_rate)
        self.class_factor = [0.06329, 0.027567, 0.90914]
        # self.class_factor = [0.1, 0.9]
        # self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.loss = self.my_loss_cat

    @tf.function(jit_compile=True)
    def step(self, x, y, train=False):

        with tf.GradientTape() as tape:
            tape.watch(self.resModel.trainable_variables)
            logits = self.resModel(x)
            smce = self.loss(y_true=y, y_pred=logits)
            if train:
                gradients = tape.gradient(smce, self.resModel.trainable_variables)

        if train:
            self.optimizer.apply_gradients(zip(gradients, self.resModel.trainable_variables))

        pred = tf.math.argmax(logits, axis=-1)
        correct_pred = tf.equal(pred, tf.math.argmax(y, axis=-1))
        correct_pred = tf.reshape(correct_pred, [-1])
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



        return smce, accuracy, logits

    def save_params(self):

        vars_to_save = {}
        for idx, name in enumerate(self.resModel.layers.name):
            vars_to_save[self.resModel.layer[idx]] = self.resModel.layers[idx]
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=self.ckpt_dir, max_to_keep=3)
        ckptman.save()

    def load_params(self):

        vars_to_load = {}
        for idx, name in enumerate(self.resModel.layers.name):
            vars_to_load[self.resModel.layer[idx]] = self.resModel.layers[idx]
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def model(self, verbose=False):

        img_input = tf.keras.layers.Input(shape=[self.height, self.width, self.channel])
        conv1 = tf.keras.layers.Conv2D(16, 3, strides=1, padding='SAME', name='Conv1')(img_input)
        conv1_act = tf.keras.layers.ELU(name='Conv1_relu')(conv1)
        convtmp_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', name='conv2_1_1')(conv1_act)
        # convtmp_1bn = tf.keras.layers.BatchNormalization(name="conv2_1_1bn")(convtmp_1)
        convtmp_1act = tf.keras.layers.ELU(name='conv2_1_1elu')(convtmp_1)
        convtmp_2 = tf.keras.layers.Conv2D(32, 3, strides=1,
                                           padding='SAME', name='conv2_1_2')(convtmp_1act)
        convtmp_2bn = tf.keras.layers.BatchNormalization(name='conv2_1_2bn')(convtmp_2)
        convtmp_2act = tf.keras.layers.ELU(name='conv2_1_2elu')(convtmp_2bn)
        conv1_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name='pool_1')(convtmp_2act)
        conv2_1 = self.residual_S(conv1_pool, ksize=self.ksize,  outchannel=64,
                                  radix=self.radix, kpaths=self.kpaths, name="conv2_1", verbose=verbose)
        conv2_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name='pool_2')(conv2_1)
        conv2_2 = self.residual_S(conv2_pool, ksize=self.ksize, outchannel=128,
                                  radix=self.radix, kpaths=self.kpaths, name="conv2_2", verbose=verbose)
        conv3_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name='pool_3')(conv2_2)
        conv3_1 = self.residual_S(conv3_pool, ksize=self.ksize, outchannel=256,
                                  radix=self.radix, kpaths=self.kpaths, name="conv3_1", verbose=verbose)
        conv4_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name='pool_4')(conv3_1)
        conv3_2 = self.residual_S(conv4_pool, ksize=self.ksize, outchannel=512,
                                  radix=self.radix, kpaths=self.kpaths, name="conv3_2", verbose=verbose)
        conv5_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name='pool_5')(conv3_2)
        conv4_1 = self.residual_S(conv5_pool, ksize=self.ksize, outchannel=512,
                                  radix=self.radix, kpaths=self.kpaths, name="conv4_1", verbose=verbose)
        conv6_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, name='pool_6')(conv4_1)

        upsample0 = self.upsample(conv6_pool, out_channel=512, apply_dropout=True, name="upsample_0")
        concat1 = tf.concat([upsample0, conv5_pool], axis=3)

        upsample1 = self.upsample(concat1, out_channel=512, apply_dropout=True, name="upsample_1")
        concat1 = tf.concat([upsample1, conv4_pool], axis=3)

        upsample2 = self.upsample(concat1, out_channel=512, apply_dropout=True, name="upsample_2")
        concat2 = tf.concat([upsample2, conv3_pool], axis=3)

        upsample3 = self.upsample(concat2, out_channel=256, name="upsample_3")
        concat3 = tf.concat([upsample3, conv2_pool], axis=3)

        upsample4 = self.upsample(concat3, out_channel=128, name="upsample_4")
        concat4 = tf.concat([upsample4, conv1_pool], axis=3)

        result = tf.keras.layers.Conv2DTranspose(self.num_class, 4, strides=2, name="f_tran", padding='same')(concat4)
        result = tf.keras.layers.Softmax()(result)

        resNest = tf.keras.Model(img_input, result)
        return resNest

    def residual_S(self, input, ksize, outchannel,
                   radix, kpaths, name="", verbose=False):
        concats_1 = None
        for idx_k in range(kpaths):
            cardinal = self.cardinal(input, ksize, outchannel // 2, radix, kpaths,
                                     name="%s_car_k%d" % (name, idx_k))
            if idx_k == 0:
                concats_1 = cardinal
            else:
                concats_1 = tf.concat([concats_1, cardinal], axis=3)
        concats_2 = tf.keras.layers.Conv2D(outchannel, ksize, strides=1, padding='SAME')(concats_1)

        if input.shape[-1] != concats_2.shape[-1]:
            convtmp_sc = tf.keras.layers.Conv2D(outchannel, 1, strides=1, padding='SAME', name="%s_cc" % name)(input)
            convtmp_scbn = tf.keras.layers.BatchNormalization(name="%s_scbn" % name)(convtmp_sc)
            convtmp_scact = tf.keras.layers.ELU(name='%s_scelu' % name)(convtmp_scbn)
            input = convtmp_scact

        output = input + concats_2

        # if verbose: print(name, output.shape)
        return output

    def cardinal(self, input, ksize, outchannel,
                 radix, kpaths, name="", verbose=False):

        if verbose: print("cardinal")
        outchannel_cv11 = int(outchannel / radix / kpaths)
        outchannel_cvkk = int(outchannel / kpaths)

        inputs = []
        for idx_r in range(radix):
            conv1 = tf.keras.layers.Conv2D(outchannel_cv11, 1, strides=1, padding='SAME',
                                           name="%s1_r%d" % (name, idx_r),)(input)
            conv1_bn = tf.keras.layers.BatchNormalization(name="%s1_%rbn" % (name, idx_r))(conv1)
            conv1_act = tf.keras.layers.ELU(name="%s1_%relu" % (name, idx_r))(conv1_bn)

            conv2 = tf.keras.layers.Conv2D(outchannel_cvkk, ksize, name="%s2_r%d" % (name, idx_r),
                                           strides=1, padding='SAME')(conv1_act)
            conv2_bn = tf.keras.layers.BatchNormalization(name="%s2_%rbn" % (name, idx_r))(conv2)
            conv2_act = tf.keras.layers.ELU(name="%s2_%relu" % (name, idx_r))(conv2_bn)
            inputs.append(conv2_act)

        return self.split_attention(inputs, outchannel_cvkk, name="%s_att" % name)

    def split_attention(self, inputs, inchannel, name="", verbose=False):

        if verbose: print("split attention")
        radix = len(inputs)
        input_holder = None
        for idx_i, input in enumerate(inputs):
            if idx_i == 0:
                input_holder = input
            else:
                input_holder += input

        ga_pool = tf.math.reduce_mean(input_holder, axis=(1, 2))
        ga_pool = tf.expand_dims(tf.expand_dims(ga_pool, axis=1), axis=1)

        dense1 = tf.keras.layers.Conv2D(inchannel // 2, 1, name="%s1" % name, strides=1, padding='SAME')(ga_pool)
        dense1_bn = tf.keras.layers.BatchNormalization(name="%s_bn" % name)(dense1)
        dense1_act = tf.keras.layers.ELU(name="%s_elu" % name)(dense1_bn)

        output_holder = None
        for idx_r in range(radix):
            dense2 = tf.keras.layers.Conv2D(inchannel, 1, name="%s2_r%d" % (name, idx_r),
                                            strides=1, padding='SAME')(dense1_act)
            if radix == 1:
                r_softmax = tf.keras.activations.sigmoid(dense2)
            elif radix > 1:
                r_softmax = tf.keras.activations.softmax(dense2)

            if idx_r == 0:
                output_holder = inputs[idx_r] * r_softmax
            else:
                output_holder += inputs[idx_r] * r_softmax

        return output_holder

    def upsample(self, inputs, out_channel, name="", apply_dropout=False):
        out = tf.keras.layers.Conv2DTranspose(out_channel, 4, strides=2,
                                     padding='same', name="%s_t_conv" % name)(inputs)

        out = tf.keras.layers.BatchNormalization()(out)

        if apply_dropout:
            out = tf.nn.dropout(out, 0.5)

        out = tf.nn.relu(out)

        return out

    # def my_loss_cat(self, y_pred, y_true):
    #     cce = tf.keras.losses.CategoricalCrossentropy()
    #     # print(y_pred.shape)
    #     p0 = cce(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    #     p1 = cce(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    #     p2 = cce(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    #     p0 = tf.math.multiply(p0, self.class_factor[0])
    #     p1 = tf.math.multiply(p1, self.class_factor[1])
    #     p2 = tf.math.multiply(p2, self.class_factor[2])
    #     # return tf.math.add(p0, p1)
    #     return tf.math.add(p0, tf.math.add(p1, p2))

    @tf.function
    def my_loss_cat(self, y_true, y_pred):
        # print(y_pred)
        # print(y_true)
        CE = 0
        for c in range(0, 3):
            scale_factor = tf.cast(tf.divide(x=1, y=tf.add(x=tf.reduce_sum(y_true[:, :, :, c], axis=0), y=1)), tf.float32)
            scale_factor = tf.divide(x=scale_factor, y=self.height*self.width)
            # CE += tf.multiply(tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            #       tf.math.log(x=tf.add(y_pred[:, :, :, c], y=1e-7)), tf.float32))), scale_factor)
            CE += tf.multiply(tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
                tf.math.log(x=tf.add(y_pred[:, :, :, c], y=1e-7)), tf.float32)), axis=0), scale_factor)
        output = tf.multiply(x=CE,y=-1)
        # print(output)
        return output


class Dataset(object):

    def __init__(self, train_path=None, val_path=None):

        print("\nInitializing Dataset...")
        train_data = np.load(train_path, allow_pickle=True)
        val_data = np.load(val_path, allow_pickle=True)

        y_tr = train_data[:, :, :, :, 0]
        y_te = val_data[:, :, :, :, 0]
        train_data = np.delete(train_data, 0, 4)
        val_data = np.delete(val_data, 0, 4)
        x_tr = np.array(train_data)
        x_te = np.array(val_data)
        x_tr = x_tr[:, :, :, :, :-1]
        x_te = x_te[:, :, :, :, :-1]
        x_tr = x_tr.reshape([-1, 256, 64, 6])
        x_te = x_te.reshape([-1, 256, 64, 6])
        y_tr = y_tr.reshape([-1, 256, 64])
        y_te = y_te.reshape([-1, 256, 64])
        # x_tr = x_tr.astype(dtype=np.float64)
        # x_te = x_te.astype(dtype=np.float64)
        # y_tr = y_tr.astype(dtype=np.float32)
        # y_te = y_te.astype(dtype=np.float32)

        class_2 = np.where(y_tr >= 1.05, y_tr - 1, 0)
        class_2 = np.where(class_2 > 1, 1, class_2)
        class_1 = np.expand_dims(np.where(y_tr > 0.95, 1 - class_2, 0), axis=3)
        class_0 = np.expand_dims(np.where(y_tr <= 0.95, 1, 0), axis=3)
        class_2 = np.expand_dims(class_2, axis=3)
        y_tr = np.concatenate((class_0, class_1, class_2), axis=3)
        y_tr = tf.convert_to_tensor(y_tr, dtype=tf.float32)
        self.x_tr, self.y_tr = x_tr, y_tr

        class_2 = np.where(y_te >= 1.05, y_te - 1, 0)
        class_2 = np.where(class_2 > 1, 1, class_2)
        class_1 = np.expand_dims(np.where(y_te > 0.95, 1 - class_2, 0), axis=3)
        class_0 = np.expand_dims(np.where(y_te <= 0.95, 1, 0), axis=3)
        class_2 = np.expand_dims(class_2, axis=3)
        y_te = np.concatenate((class_0, class_1, class_2), axis=3)
        y_te = tf.convert_to_tensor(y_te, dtype=tf.float32)
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
        try:
            self.channel = x_sample.shape[2]
        except:
            self.channel = 1

        self.min_val, self.max_val = x_sample.min(), x_sample.max()
        # self.num_class = int(np.floor(y_te.max()+1))
        self.num_class = 3

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" % (self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" % (self.min_val, self.max_val))
        print("Class  %d" % self.num_class)

    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    # Currently not in use. Moved to the dataset init to save processing time.
    def label2vector(self, labels):
        # label_out = np.eye(self.num_class)[labels]

        # Attempt with varying values.
        class_2 = np.where(labels >= 1.05, labels - 1, 0)
        class_2 = np.where(class_2 > 1, 1, class_2)
        class_1 = np.expand_dims(np.where(labels > 0.95, 1 - class_2, 0), axis=3)
        class_0 = np.expand_dims(np.where(labels <= 0.95, 1, 0), axis=3)
        class_2 = np.expand_dims(class_2, axis=3)

        # Attempt with one hot vectors
        # class_2 = np.expand_dims(np.where(labels > 1.05, 1, 0), axis=3)
        # class_1 = np.expand_dims(np.where((labels > 0.95) & (labels < 1.05), 1, 0), axis=3)
        # class_0 = np.expand_dims(np.where(labels < 0.95, 1, 0), axis=3)

        label_out = np.concatenate((class_0, class_1, class_2), axis=3)
        # print(label_out[0, 110:112, :, :])
        label_out = tf.convert_to_tensor(label_out, dtype=tf.float32)
        return label_out

    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size
        x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]

        terminator = False
        if end >= self.num_tr:
            terminator = True
            self.idx_tr = 0
            # self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else: self.idx_tr = end

        if fix: self.idx_tr = start

        if x_tr.shape[0] != batch_size:
            x_tr, y_tr = self.x_tr[-1-batch_size:-1], self.y_tr[-1-batch_size:-1]

        return x_tr, y_tr, terminator

    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te + batch_size
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]

        terminator = False
        if end >= self.num_te:
            terminator = True
            self.idx_te = 0
        else: self.idx_te = end

        if x_te.shape[0] != batch_size:
            x_te, y_te = self.x_te[-1-batch_size:-1], self.y_te[-1-batch_size:-1]
        return x_te, y_te, terminator


def training(neuralnet, dataset, epochs, batch_size):
    print("\nTraining to %d epochs (%d of minibatch size)" % (epochs, batch_size))

    iteration = 0
    f1cur = 0
    f1prev = 0
    # tf.keras.utils.plot_model(neuralnet.resModel, to_file='ResNeSt.png', show_shapes=True)

    for epoch in range(epochs):
        if neuralnet.learning_rate < 1e-5:
            break
        while True:
            x_tr, y_tr, terminator = dataset.next_train(batch_size)  # y_tr does not used in this prj.
            loss, accuracy, class_score = neuralnet.step(x=x_tr, y=y_tr, train=True)
            loss = tf.reduce_sum(loss)
            iteration += 1
            print('.', end='')
            if (iteration + 1) % 100 == 0:
                print()
            if terminator: break

            # neuralnet.save_params()
        print()
        print("Epoch [%d / %d] (%d iteration)  Loss:%.5f, Acc:%.5f"
              % (epoch, epochs, iteration, loss, accuracy))
        if epoch % 3 == 0:
            f1cur = test(neuralnet, dataset, epoch)
        if f1cur < f1prev:
            neuralnet.learning_rate /= 5
            print("New Learning Rate = {}".format(neuralnet.learning_rate))
        f1prev = f1cur


def test(neuralnet, dataset, epoch):
    print("\nTest...")

    total_loss = 0
    while True:
        x_te, y_te, terminator = dataset.next_test(1)  # y_te does not used in this prj.
        loss, accuracy, class_score = neuralnet.step(x=x_te, y=y_te, train=False)
        loss = tf.reduce_sum(loss)
        precision.update_state(y_te, class_score)
        recall.update_state(y_te, class_score)
        pre_c2.update_state(y_te[:, :, -1], class_score[:, :, -1])
        re_c2.update_state(y_te[:, :, -1], class_score[:, :, -1])
        total_loss += loss
        if terminator: break

    total_loss /= dataset.num_te
    with summary_writer.as_default():
        tf.summary.scalar("loss", total_loss, step=epoch)
        print("loss = {}".format(total_loss))
        f1 = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
        tf.summary.scalar("val_f1", f1, step=epoch)
        print("f1 = {}".format(f1))
        tf.summary.scalar("val_precision", precision.result(), step=epoch)
        print("precision = {}".format(precision.result()))
        precision.reset_states()
        tf.summary.scalar("recall_recall", recall.result(), step=epoch)
        print("recall = {}".format(recall.result()))
        precision.reset_states()

        f1_2 = 2 * (pre_c2.result() * re_c2.result()) / (pre_c2.result() + re_c2.result())
        tf.summary.scalar("c2_f1", f1_2, step=epoch)
        tf.summary.scalar("c2_precision", pre_c2.result(), step=epoch)
        precision.reset_states()
        tf.summary.scalar("c2_recall", re_c2.result(), step=epoch)
        precision.reset_states()

        tf.summary.scalar("loss", total_loss, step=epoch)
        return f1


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_data = '/DATA/TBI/Datasets/NPFiles/DispBal/TrainingData.npy'
    val_data = '/DATA/TBI/Datasets/NPFiles/DispBal/ValidationData.npy'
    dataset = Dataset(train_data, val_data)
    # config = tf.estimator.RunConfig(train_distribute=mirrored_strategy)
    neuralnet = ResNest(height=dataset.height, width=dataset.width, channel=dataset.channel,
                        num_class=dataset.num_class, radix=3, ksize=3, learning_rate=5e-3)
    # classifier = tf.estimator.Estimator(
    #     model_fn=neuralnet, model_dir='/tmp/multiworker', config=config)
    # tf.estimator.train_and_evaluate(
    #     classifier,
    #     train_spec=tf.estimator.TrainSpec(input_fn=training(neuralnet=neuralnet, dataset=dataset, epochs=76, batch_size=64))
    # )

    training(neuralnet=neuralnet, dataset=dataset, epochs=46, batch_size=64)
    # test(neuralnet=neuralnet, dataset=dataset)
    neuralnet.resModel.save('/DATA/TBI/Datasets/Models/ResNeSt_D1')


if __name__ == '__main__':

    main()
