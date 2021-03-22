import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

training_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/Polar3Class/TrainingData.npy'
training_label_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/Polar3Class/TrainingLabels.npy'
testing_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/Polar3Class/ValidationData.npy'
testing_label_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/Polar3Class/ValidationLabels.npy'
validation_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/Polar3Class/ValidationData.npy'
validation_label_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/Polar3Class/ValidationLabels.npy'

# channel 0: outside the brain
# channel 1: no-bleed
# channel 2: bleed
OUTPUT_CHANNELS = 3
BATCH_SIZE = 50
BUFFER_SIZE = 100
BLEED_THRESHOLD = .015
xdim = 256
ydim = 64
bleedCount = 0
nonBleedCount = 0

x_train = np.load(training_data_path)
x_train = x_train[:, :, :, :, range(1, 18)]
y_train = np.load(training_label_path)
x_test = np.load(testing_data_path)
x_test = x_test[:, :, :, :, range(1, 18)]
y_test = np.load(testing_label_path)
x_val = np.load(validation_data_path)
x_val = x_val[:, :, :, :, range(1, 18)]
y_val = np.load(validation_label_path)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
y_val = y_val.astype(int)


x_shape = x_train.shape
y_shape = x_test.shape
z_shape = x_val.shape
x_train = np.reshape(x_train, (x_shape[0], x_shape[2], x_shape[3], x_shape[4]))
x_test = np.reshape(x_test, (y_shape[0], y_shape[2], y_shape[3], y_shape[4]))
x_val = np.reshape(x_val, (z_shape[0], z_shape[2], z_shape[3], z_shape[4]))

# print(y_train)
# print(y_test)
# print(x_train.shape)
# print(x_test.shape)


def downsample(filters, size, conv_id, stride=2, batch_norm=True):
  initializer = tf.random_normal_initializer(0., 0.2)

  result = keras.Sequential()  # construct Sequential model
  result.add(
      keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                          kernel_initializer=initializer, use_bias=False,
                          name='conv_{}'.format(conv_id)))

  if batch_norm:
    result.add(keras.layers.BatchNormalization())

  result.add(keras.layers.LeakyReLU())
  return result


def SegNet():
  inputs = keras.layers.Input(shape=[xdim, ydim, 17])

  fScaleFactor = 8
  # encoder layers
  down_stack = [
    downsample(4 * fScaleFactor,  4, conv_id=0, batch_norm=False),  # (bs, 128, 32, 64)
    downsample(16 * fScaleFactor, 4, conv_id=1),  # (bs, 64, 16, 256)
    downsample(32 * fScaleFactor, 4, conv_id=2),  # (bs, 32, 8, 512)
    downsample(32 * fScaleFactor, 4, conv_id=3),  # (bs, 16, 4, 512)
    downsample(32 * fScaleFactor, 4, conv_id=4),  # (bs, 8, 2, 512)
    downsample(32 * fScaleFactor, 4, conv_id=5),  # (bs, 4, 1, 512)
  ]

  up_stack = [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(1, activation='sigmoid'),
  ]

  x = inputs

  # Downsampling through the model
  # iterate over the downsample layers and connect them together
  for down in down_stack:
    x = down(x)

  # Upsampling and establishing the skip connections
  for up in up_stack:
    x = up(x)

  return keras.Model(inputs=inputs, outputs=x)


stop_callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-3 less"
        min_delta=1e-4,
        # "no longer improving" being further defined as "for at least 7 epochs"
        patience=5,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]

SegNetF = SegNet()

# print(SegNetF.summary())
tf.keras.utils.plot_model(SegNetF, to_file='NNTBI.png', show_shapes=True)

epochs = 30

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
segnet_optimizer = keras.optimizers.Adam(1e-5, beta_1=0.5)

SegNetF.compile(optimizer=segnet_optimizer,
                loss=keras.losses.BinaryCrossentropy(),
                metrics=['mse'])

SegNetF.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_data=(x_test, y_test),
            epochs=epochs,
            callbacks=[tensorboard_callback]
            )

for image, label in zip(x_test, y_test):
    image = np.reshape(image, (1, 256, 64, 17))
    pred = SegNetF(image)
    # pred = tf.math.round(pred)
    tf.print(pred, label)

print("End of testing set \n")

for image, label in zip(x_val, y_val):
    image = np.reshape(image, (1, 256, 64, 17))
    pred = SegNetF(image)
    # pred = tf.math.round(pred)
    tf.print(pred, label)

SegNetF.save('nn_polar_1')
