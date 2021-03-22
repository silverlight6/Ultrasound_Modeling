import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

training_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/TrainingPolarData.npy'
testing_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/TestingPolarData.npy'

OUTPUT_CHANNELS = 3
BATCH_SIZE = 5
BUFFER_SIZE = 100
xdim = 256
ydim = 64


def preProcess(input_data):
    t_y = tf.gather(input_data, 0, axis=3)
    t_x = tf.gather(input_data, list(range(1, 15)), axis=3)
    t_y = tf.cast(t_y, dtype=tf.int32)
    # t_y = tf.reshape(t_y, [1, xdim * ydim, 1])
    t_y = tf.one_hot(t_y, depth=OUTPUT_CHANNELS)
    return t_x, t_y


train_data = tf.data.Dataset.from_tensor_slices(np.load(training_data_path))
test_data = tf.data.Dataset.from_tensor_slices(np.load(testing_data_path))
train_data = train_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_data.shuffle(BUFFER_SIZE)
train_data.batch(BATCH_SIZE)
test_data.shuffle(BUFFER_SIZE)
test_data.batch(BATCH_SIZE)

image_shape = [xdim, ydim, 14]

generator = tf.keras.models.load_model("tbi_segnet_polar_0.h5")

generator_optimizer = tf.keras.optimizers.Adam(2e-1, beta_1=0.5)

epochs = 30

# class_weights = {0: tf.cast(tf.fill([xdim * ydim], .1), dtype=tf.float32),
#                  1: tf.cast(tf.fill([xdim * ydim], 1), dtype=tf.float32),
#                  2: tf.cast(tf.fill([xdim * ydim], 10), dtype=tf.float32)}


@tf.function
def my_precision(y_true, y_pred):
    metric = tf.keras.metrics.Precision()
    return metric(y_true[:, :, :, 2], y_pred[:, :, :, 2])


@tf.function
def my_recall(y_true, y_pred):
    metric = tf.keras.metrics.Recall()
    return metric(y_true[:, :, :, 2], y_pred[:, :, :, 2])


@tf.function
def my_cat_cr_entropy(y_true, y_pred):
    metric = tf.keras.metrics.CategoricalCrossentropy()
    return metric(y_true[:, :, :, 2], y_pred[:, :, :, 2])


def my_loss_cat(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    p0 = cce(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    p1 = cce(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    p2 = cce(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    p0 = tf.math.multiply(p0, 0.1)
    p1 = tf.math.multiply(p1, 0.5)
    p2 = tf.math.multiply(p2, 10)
    return tf.math.add(p0, tf.math.add(p1, p2))


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

stop_callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-3 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 7 epochs"
        patience=10,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

generator.compile(optimizer=generator_optimizer,
                  loss=my_loss_cat,
                  metrics=['Precision', 'Recall', my_precision, my_recall, my_cat_cr_entropy])

generator.fit(train_data,
              batch_size=BATCH_SIZE,
              shuffle=True,
              validation_data=test_data,
              epochs=epochs,
              callbacks=[tensorboard_callback, stop_callbacks])

generator.save('tbi_segnet_polar_3.h5')
