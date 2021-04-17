import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime


def main():
    training_data_path = '/DATA/TBI/Datasets/NPFiles/CardiacBalanced/TrainingData2.npy'
    testing_data_path = '/DATA/TBI/Datasets/NPFiles/CardiacBalanced/ValidationData2.npy'

    # channel 0: outside the brain
    # channel 1: no-bleed
    # channel 2: bleed
    OUTPUT_CHANNELS = 3
    BATCH_SIZE = 30
    BUFFER_SIZE = 100
    xdim = 256
    ydim = 64
    pct = [0, 0, 0]  # Positive Count Total Class 0 or Percentage per Class Total
    class_factor = [0.06725, 0.03851, 0.89423]
    # class_factor = [0.3, 0.1, 0.6]

    train_data = np.load(training_data_path)
    val_data = np.load(testing_data_path)

    y_tr = train_data[:, :, :, :, 0]
    y_te = val_data[:, :, :, :, 0]
    train_data = np.delete(train_data, 0, 4)
    val_data = np.delete(val_data, 0, 4)
    x_tr = np.array(train_data)
    x_te = np.array(val_data)
    y_tr = y_tr.astype(dtype=np.int32)
    y_te = y_te.astype(dtype=np.int32)
    y_tr = np.eye(OUTPUT_CHANNELS)[y_tr]
    y_te = np.eye(OUTPUT_CHANNELS)[y_te]
    y_tr = y_tr[:, 0, :, :, :]
    y_te = y_te[:, 0, :, :, :]
    y_tr = y_tr.reshape([-1, 256, 64, 3])
    y_te = y_te.reshape([-1, 256, 64, 3])
    x_tr = x_tr.astype(dtype=np.float64)
    x_te = x_te.astype(dtype=np.float64)
    print(x_tr.shape)
    x_tr0, x_tr1, x_tr2, x_tr3, x_tr4, x_tr5, x_tr6, x_tr7, x_tr8, x_tr9 = [], [], [], [], [], [], [], [], [], [],
    for i in range(0, x_tr.shape[0]):
        x_tr0 = x_tr[:, 0, :, :, :]
        x_tr1 = x_tr[:, 1, :, :, :]
        x_tr2 = x_tr[:, 2, :, :, :]
        x_tr3 = x_tr[:, 3, :, :, :]
        x_tr4 = x_tr[:, 4, :, :, :]
        x_tr5 = x_tr[:, 5, :, :, :]
        x_tr6 = x_tr[:, 6, :, :, :]
        x_tr7 = x_tr[:, 7, :, :, :]
        x_tr8 = x_tr[:, 8, :, :, :]
        x_tr9 = x_tr[:, 9, :, :, :]

    x_te0, x_te1, x_te2, x_te3, x_te4, x_te5, x_te6, x_te7, x_te8, x_te9 = [], [], [], [], [], [], [], [], [], [],

    for i in range(0, x_te.shape[0]):
        x_te0 = x_te[:, 0, :, :, :]
        x_te1 = x_te[:, 1, :, :, :]
        x_te2 = x_te[:, 2, :, :, :]
        x_te3 = x_te[:, 3, :, :, :]
        x_te4 = x_te[:, 4, :, :, :]
        x_te5 = x_te[:, 5, :, :, :]
        x_te6 = x_te[:, 6, :, :, :]
        x_te7 = x_te[:, 7, :, :, :]
        x_te8 = x_te[:, 8, :, :, :]
        x_te9 = x_te[:, 9, :, :, :]

    print(x_te.shape)

    # Code for finding class distributions
    pct[0] += tf.math.reduce_sum(y_tr[:, :, :, 0])
    pct[1] += tf.math.reduce_sum(y_tr[:, :, :, 1])
    pct[2] += tf.math.reduce_sum(y_tr[:, :, :, 2])

    print("Initial pct[0] = {}, pct[1] = {}, pct[2] = {},".format(pct[0], pct[1], pct[2]))

    pct = tf.cast(pct, dtype=tf.float64)
    totalPos = pct[0] + pct[1] + pct[2]
    pct = tf.math.divide(pct, totalPos)

    print("totalPos = {}".format(totalPos))
    print("Normalized 1 pct[0] = {}, pct[1] =  {}, pct[2] = {},".format(pct[0], pct[1], pct[2]))

    pct = tf.math.divide(1, pct)
    pct = tf.math.divide(pct, 3)

    print("Inverse 1 pct[0] = {}, pct[1] =  {}, pct[2] = {},".format(pct[0], pct[1], pct[2]))
    totalPos = pct[0] + pct[1] + pct[2]
    pct = tf.math.divide(pct, totalPos)

    print("Normalized 2 pct[0] = {}, pct[1] = {}, pct[2] = {},".format(pct[0], pct[1], pct[2]))


    def downsample(filters, size, conv_id, stride=2, batch_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = keras.Sequential()  # construct Sequential model
        result.add(
            keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                                kernel_initializer=initializer, use_bias=False,
                                name='conv_{}'.format(conv_id)))

        if batch_norm:
            result.add(keras.layers.BatchNormalization())
        result.add(keras.layers.LeakyReLU())
        return result


    def upsample(filters, size, apply_dropout=False):
        initializer = keras.initializers.RandomNormal(0., 0.02)

        result = keras.Sequential()
        result.add(
            keras.layers.Conv2DTranspose(filters, size, strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False))

        result.add(keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(keras.layers.Dropout(0.5))

        result.add(keras.layers.ReLU())

        return result


    def Mask_Gen():
        imp0 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp0")
        imp1 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp1")
        imp2 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp2")
        imp3 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp3")
        imp4 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp4")
        imp5 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp5")
        imp6 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp6")
        imp7 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp7")
        imp8 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp8")
        imp9 = tf.keras.layers.Input(shape=[xdim, ydim, 3], name="imp9")

        inputConv0 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_0')(imp0)
        inputConv1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_1')(imp1)
        inputConv2 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_2')(imp2)
        inputConv3 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_3')(imp3)
        inputConv4 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_4')(imp4)
        inputConv5 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_5')(imp5)
        inputConv6 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_6')(imp6)
        inputConv7 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_7')(imp7)
        inputConv8 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_8')(imp8)
        inputConv9 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', use_bias=False,
                                            name='inpConv_9')(imp9)

        input_act0 = tf.keras.layers.ELU(name='inpELU_0')(inputConv0)
        input_act1 = tf.keras.layers.ELU(name='inpELU_1')(inputConv1)
        input_act2 = tf.keras.layers.ELU(name='inpELU_2')(inputConv2)
        input_act3 = tf.keras.layers.ELU(name='inpELU_3')(inputConv3)
        input_act4 = tf.keras.layers.ELU(name='inpELU_4')(inputConv4)
        input_act5 = tf.keras.layers.ELU(name='inpELU_5')(inputConv5)
        input_act6 = tf.keras.layers.ELU(name='inpELU_6')(inputConv6)
        input_act7 = tf.keras.layers.ELU(name='inpELU_7')(inputConv7)
        input_act8 = tf.keras.layers.ELU(name='inpELU_8')(inputConv8)
        input_act9 = tf.keras.layers.ELU(name='inpELU_9')(inputConv9)

        img_input = input_act0 + input_act1 + input_act2 + input_act3 + input_act4 + input_act5 + input_act6 + input_act7 + input_act8 + input_act9

        fScaleFactor = 8
        # encoder layers
        down_stack = [
            downsample(4 * fScaleFactor, 4, conv_id=0, batch_norm=False),  # (bs, 128, 32, 64)
            downsample(16 * fScaleFactor, 4, conv_id=1),  # (bs, 64, 16, 256)
            downsample(32 * fScaleFactor, 4, conv_id=2),  # (bs, 32, 8, 512)
            downsample(32 * fScaleFactor, 4, conv_id=3),  # (bs, 16, 4, 512)
            downsample(32 * fScaleFactor, 4, conv_id=4),  # (bs, 8, 2, 512)
            downsample(32 * fScaleFactor, 4, conv_id=5),  # (bs, 4, 1, 512)
        ]

        # decoder layers
        up_stack = [
            upsample(32 * fScaleFactor, 4, apply_dropout=True),  # (bs, 4, 1, 1024)
            upsample(32 * fScaleFactor, 4, apply_dropout=True),  # (bs, 8, 2, 1024)
            upsample(32 * fScaleFactor, 4, apply_dropout=True),  # (bs, 16, 4, 1024)
            upsample(16 * fScaleFactor, 4),  # (bs, 32, 8, 768)
            upsample(8 * fScaleFactor, 4),  # (bs, 64, 16, 640)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        # "deconvolutional" operation, enlarging/expanding the image
        last = keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='softmax')  # (bs, 256, 64, 4)

        x = img_input

        # Downsampling through the model
        skips = []
        # iterate over the downsample layers and connect them together
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = keras.layers.Concatenate()([x, skip])

        x = last(x)
        # x = softLayer(x)

        return keras.Model(inputs=[imp0, imp1, imp2, imp3, imp4, imp5, imp6, imp7, imp8, imp9], outputs=x)


    mask_gen = Mask_Gen()

    print(mask_gen.summary())
    # tf.keras.utils.plot_model(mask_gen, to_file='SegNet.png', show_shapes=True)

    mask_gen_optimizer = keras.optimizers.Adam(2e-3)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    stop_callbacks = [
        keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-3 less"
            min_delta=1e-4,
            # "no longer improving" being further defined as "for at least 7 epochs"
            patience=7,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]

    epochs = 50


    # EXPERIMENTAL CODE HERE
    # The purpose of these things is to make the network pay more attention to the bleed and for us to be able to track how.
    # ATTENTION it pays towards that bleed.
    def my_loss_cat(y_true, y_pred):
        CE = 0
        for c in range(0, OUTPUT_CHANNELS):
            scale_factor = 1 / (tf.reduce_sum(y_true[:, :, :, c]) + 1)
            CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
                tf.math.log(y_pred[:, :, :, c] + 1e-7), tf.float32))) * scale_factor * class_factor[c]
        return CE * -1


    def my_loss_cat_1(y_true, y_pred):
        CE = 0
        cce = tf.keras.losses.CategoricalCrossentropy()
        for c in range(0, OUTPUT_CHANNELS):
            CE += cce(y_pred=y_pred, y_true=y_true) * class_factor[c]
        return CE * OUTPUT_CHANNELS


    @tf.function
    class CatRecall(keras.metrics.Metric):
        def __init__(self, name="bleed_Recall", **kwargs):
            super(CatRecall, self).__init__(name=name, **kwargs)
            self.recall = self.add_weight(name="ctp", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = y_pred[:, :, :, 2]
            tf.math.round(y_pred)
            y_true = y_true[:, :, :, 2]
            truePos = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
            truePos = tf.cast(y_true, "int32") == tf.cast(truePos, "int32")
            truePos = tf.cast(truePos, "float32")
            truePosCount = tf.reduce_sum(truePos)
            totalPosCount = tf.reduce_sum(y_true)
            self.recall.assign_add(tf.math.divide(truePosCount, totalPosCount))

        def result(self):
            return self.recall

        def reset_states(self):
            # The state of the metric will be reset at the start of each epoch.
            self.recall.assign(0.0)

    mask_gen.compile(optimizer=mask_gen_optimizer,
                     # loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                     loss=my_loss_cat_1,
                     metrics=['Recall', 'Precision'])

    mask_gen.fit({"imp0": x_tr0, "imp1": x_tr1, "imp2": x_tr2, "imp3": x_tr3, "imp4": x_tr4,
                  "imp5": x_tr5, "imp6": x_tr6, "imp7": x_tr7, "imp8": x_tr8, "imp9": x_tr9},
                 y_tr,
                 shuffle=True,
                 validation_data=({"imp0": x_te0, "imp1": x_te1, "imp2": x_te2, "imp3": x_te3, "imp4": x_te4,
                                   "imp5": x_te5, "imp6": x_te6, "imp7": x_te7, "imp8": x_te8, "imp9": x_te9}, y_te),
                 epochs=epochs,
                 callbacks=[tensorboard_callback]
                 )

    mask_gen.save('/DATA/TBI/Datasets/Models/tbi_segnet_2_0')


if __name__ == '__main__':

    main()

