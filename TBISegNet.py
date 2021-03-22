import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

training_data_path = '/data/TBI/Datasets/NPFiles/Cardiac/TrainingData.npy'
testing_data_path = '/data/TBI/Datasets/NPFiles/Cardiac/ValidationData.npy'

# channel 0: outside the brain
# channel 1: no-bleed
# channel 2: bleed
OUTPUT_CHANNELS = 3
BATCH_SIZE = 50
BUFFER_SIZE = 100
xdim = 256
ydim = 64
pct = [0, 0, 0]  # Positive Count Total Class 0 or Percentage per Class Total
class_factor = [0.06725, 0.03851, 0.89423]


def preProcess(input_data):
    print(input_data)
    t_y = tf.gather(input_data, 0, axis=3)  # weeding out the labels
    t_x = tf.gather(input_data, list(range(1, 16)), axis=3)  # weeding out the x data
    t_y = tf.cast(t_y, dtype=tf.int32)  # choose int32 types for the data
    t_y = tf.one_hot(t_y, depth=OUTPUT_CHANNELS)  # convert to 3 bits to represent classes
    tf.debugging.check_numerics(t_x, "x contains Nan")
    tf.debugging.check_numerics(t_x, "y contains Nan")
    return t_x, t_y  # return input and output


# load the numpy arrays into TensorFlow dataset object
train_data = tf.data.Dataset.from_tensor_slices(np.load(training_data_path))
test_data = tf.data.Dataset.from_tensor_slices(np.load(testing_data_path))
# use map to call the function "preProcess" on each training item and testing item
train_data = train_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_data.shuffle(BUFFER_SIZE)  # shuffle the data
train_data.batch(BATCH_SIZE)  # make data into batches, based on batch size

print(train_data)
test_data.batch(BATCH_SIZE)  # make data into batches, based on batch size

image_shape = [xdim, ydim, 15]

# Code for finding class distributions
for _, label in train_data:
    pct[0] += tf.math.reduce_sum(label[:, :, :, 0])
    pct[1] += tf.math.reduce_sum(label[:, :, :, 1])
    pct[2] += tf.math.reduce_sum(label[:, :, :, 2])

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
    inputs = keras.layers.Input(shape=[xdim, ydim, 15])

    fScaleFactor = 8
    # encoder layers
    down_stack = [
        downsample(4 * fScaleFactor, 4, conv_id=0, batch_norm=False),  # (bs, 128, 32, 64)
        downsample(16 * fScaleFactor, 4, conv_id=1, batch_norm=False),  # (bs, 64, 16, 256)
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
        upsample(16 * fScaleFactor, 4, apply_dropout=True),  # (bs, 32, 8, 768)
        upsample(8 * fScaleFactor, 4),  # (bs, 64, 16, 640)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    # "deconvolutional" operation, enlarging/expanding the image
    last = keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        activation='softmax')  # (bs, 256, 64, 4)

    x = inputs

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

    return keras.Model(inputs=inputs, outputs=x)


mask_gen = Mask_Gen()

print(mask_gen.summary())
# tf.keras.utils.plot_model(mask_gen, to_file='SegNet.png', show_shapes=True)

mask_gen_optimizer = keras.optimizers.Adam(2e-2, beta_1=0.5)

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
    for c in range(0, 3):
        scale_factor = 1 / (tf.reduce_sum(y_true[:, :, :, c]) + 1)
        # tf.print(y_pred[:, :, :, c])
        CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            tf.math.log(y_pred[:, :, :, c]), tf.float32))) * scale_factor * class_factor[c]
    return CE * -1


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
                 loss=my_loss_cat,
                 metrics=['Recall', 'Precision'])

mask_gen.fit(train_data,
             shuffle=True,
             validation_data=test_data,
             epochs=epochs,
             callbacks=[tensorboard_callback, stop_callbacks]
             )

mask_gen.save('/data/TBI/Datasets/Models/tbi_segnet_0')
