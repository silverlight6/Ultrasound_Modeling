import tensorflow as tf
import numpy as np
from tensorflow import keras
from datetime import datetime
from matplotlib import pyplot as plt

training_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/PolarTimeData/TrainingData.npy'
testing_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/PolarTimeData/ValidationData.npy'


# channel 0: outside the brain
# channel 1: no-bleed
# channel 2: bleed
OUTPUT_CHANNELS = 3
BATCH_SIZE = 20
BUFFER_SIZE = 100
xdim = 240
ydim = 80
class_factor = [1.475, 0.678, 7.847]


def preProcess(input_data):
    t_y = tf.gather(input_data, 0, axis=3)                   # weeding out the labels
    t_x = tf.gather(input_data, list(range(1, 4)), axis=3)  # weeding out the x data
    t_y = tf.cast(t_y, dtype=tf.int32)                       # choose int32 types for the data
    t_y = tf.one_hot(t_y, depth=OUTPUT_CHANNELS)             # convert to 3 bits to represent classes
    return t_x, t_y                                          # return input and output


# load the numpy arrays into TensorFlow dataset object
train_data = tf.data.Dataset.from_tensor_slices(np.load(training_data_path))
test_data = tf.data.Dataset.from_tensor_slices(np.load(testing_data_path))
# use map to call the function "preProcess" on each training item and testing item
train_data = train_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_data.shuffle(BUFFER_SIZE)  # shuffle the data
train_data.batch(BATCH_SIZE)     # make data into batches, based on batch size

print(train_data)
test_data.batch(BATCH_SIZE)      # make data into batches, based on batch size

image_shape = [xdim, ydim, 3]


def upsample(filters, size, apply_dropout=False):
  initializer = keras.initializers.RandomNormal(0., 0.02)

  result = keras.Sequential()
  result.add(
    keras.layers.Conv2DTranspose(filters, size, strides=2,
                                 padding='same',
                                 kernel_initializer=initializer,
                                 use_bias=False))

  # result.add(keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(keras.layers.Dropout(0.5))

  result.add(keras.layers.LeakyReLU())

  return result


base_model = tf.keras.applications.InceptionV3(input_shape=[xdim, ydim, 3], include_top=False)


def Mask_Gen():
  inputs = keras.layers.Input(shape=[xdim, ydim, 3])

  # print(base_model.summary())
  print(len(base_model.layers))

  # Use the activations of these layers
  layer_names = [
      'activation_2',  # 117x37
      'activation_4',  # 56x16
      'mixed2',  # 27x7
      'mixed7',  # 13x3
      'mixed10'  # 6x1
  ]

  layers = [base_model.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

  down_stack.trainable = False
  # down_stack.trainable = True

  # decoder layers
  up_stack = [
    upsample(256, 4, apply_dropout=True),  # (bs, 6, 1, 1024)
    upsample(256, 4, apply_dropout=True),  # (bs, 13, 3, 1024)
    upsample(256, 4, apply_dropout=True),  # (bs, 27, 7, 1024)
    upsample(128, 4),  # (bs, 56, 16, 768)
    upsample(64, 4),  # (bs, 117, 37, 640)
  ]

  padding_stack = [
      keras.layers.ZeroPadding2D(((0, 1), (1, 0))),
      keras.layers.ZeroPadding2D(((0, 1), (1, 0))),
      keras.layers.ZeroPadding2D((1, 1)),
      keras.layers.ZeroPadding2D(((2, 3), (3, 2))),
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  # "deconvolutional" operation, enlarging/expanding the image
  last = keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      activation='softmax')  # (bs, 256, 64, 4)

  x = inputs

  # x = keras.layers.Conv2D(3, 5, strides=1, padding='same',
  #                         use_bias=False)(x)

  # iterate over the downsample layers and connect them together
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip, pad in zip(up_stack, skips, padding_stack):
    x = up(x)
    x = pad(x)
    x = keras.layers.Concatenate()([x, skip])

  x = keras.layers.ZeroPadding2D(((1, 2), (2, 1)))(x)
  x = last(x)

  return keras.Model(inputs=inputs, outputs=x)


seg_model = Mask_Gen()

# print(seg_model.summary())
seg_model_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

epochs = 30

tf.keras.utils.plot_model(seg_model, to_file='inception_total.png', show_shapes=True)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

stop_callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        min_delta=1e-4,
        patience=7,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
]


def generate_images(model, test_input, target):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [target[0], prediction[0]]
  title = ['Ground Truth', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i][:, :, -1])
    plt.axis('off')
  plt.show()


def my_loss_cat(y_true, y_pred):
    CE = 0.0
    scale_factor = 1 / tf.reduce_sum(y_true[:, :, :, :])
    for c in range(0, OUTPUT_CHANNELS):
        # print(y_pred[:, :, :, c])
        CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            tf.math.log(y_pred[:, :, :, c]), tf.float64))) * scale_factor * class_factor[c]
    return CE * -1 * OUTPUT_CHANNELS


seg_model.compile(optimizer=seg_model_optimizer,
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy', 'Recall'])

seg_model.fit(train_data,
              batch_size=BATCH_SIZE,
              shuffle=True,
              validation_data=test_data,
              epochs=epochs,
              callbacks=[tensorboard_callback, stop_callbacks])

for testx, testy in test_data.take(10):
    generate_images(seg_model, testx, testy)

seg_model_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
base_model.trainable = True
seg_model.trainable = True

seg_model.compile(optimizer=seg_model_optimizer,
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy', 'Recall'])

seg_model.fit(train_data,
              batch_size=BATCH_SIZE,
              shuffle=True,
              validation_data=test_data,
              epochs=epochs,
              callbacks=[tensorboard_callback, stop_callbacks])

seg_model.save('tbi_seg_transfer.h5')
base_model.save('inception')

for testx, testy in test_data.take(10):
    generate_images(seg_model, testx, testy)
