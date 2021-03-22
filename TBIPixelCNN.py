import tensorflow as tf
import numpy as np
import time
import datetime

from IPython import display
from matplotlib import pyplot as plt

training_data_path = '/data/TBI/Datasets/NPFiles/IPH/TrainingData.npy'
testing_data_path = '/data/TBI/Datasets/NPFiles/IPH/ValidationData.npy'

OUTPUT_CHANNELS = 3
BATCH_SIZE = 256
BUFFER_SIZE = 100
image_shape = [256, 64, 15]
class_factor = [0.06329, 0.027567, 0.90914]


def preProcess(input_data):
    t_y = tf.gather(input_data, 0, axis=3)         # weeding out the labels
    t_x = tf.gather(input_data, list(range(1, 16)), axis=3)  # weeding out the x data
    t_y = tf.cast(t_y, dtype=tf.int32)  # choose int32 types for the data
    t_y = tf.one_hot(t_y, depth=OUTPUT_CHANNELS)  # convert to 3 bits to represent classes
    tf.debugging.check_numerics(t_x, "x contains Nan")
    tf.debugging.check_numerics(t_x, "y contains Nan")
    return t_x, t_y                                          # return input and output


train_data = tf.data.Dataset.from_tensor_slices(np.load(training_data_path))
test_data = tf.data.Dataset.from_tensor_slices(np.load(testing_data_path))
train_data = train_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_data.shuffle(BUFFER_SIZE)
train_data.batch(BATCH_SIZE)

test_data.shuffle(BUFFER_SIZE)
test_data.batch(BATCH_SIZE)


def SMobileNetV2(input_shape):

    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.ZeroPadding2D(padding=1, name='Conv1_pad')(img_input)
    x = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=(2, 2),
                               padding='valid',
                               use_bias=True,
                               name='Conv1')(x)
    x = tf.keras.layers.LeakyReLU(6., name='Conv1_relu')(x)
    x = _inverted_res_block(x, filters=16, in_filters=32, stride=1,
                            expansion=6, block_id=0)
    x = _inverted_res_block(x, filters=24, in_filters=16, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, in_filters=24, stride=1,
                            expansion=6, block_id=2)
    x = _inverted_res_block(x, filters=32, in_filters=24, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, in_filters=32, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, in_filters=32, stride=1,
                            expansion=6, block_id=5)
    x = _inverted_res_block(x, filters=64, in_filters=32, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, in_filters=64, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, in_filters=64, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, in_filters=64, stride=1,
                            expansion=6, block_id=9)
    x = _inverted_res_block(x, filters=96, in_filters=64, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, in_filters=96, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, in_filters=96, stride=2,
                            expansion=6, block_id=12)
    x = _inverted_res_block(x, filters=160, in_filters=96, stride=1,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, in_filters=160, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, in_filters=160, stride=1,
                            expansion=6, block_id=15)

    last_block_filters = 1280

    x = tf.keras.layers.Conv2D(last_block_filters,
                               kernel_size=1,
                               use_bias=False,
                               name='Conv_2')(x)
    x = tf.keras.layers.LeakyReLU(6., name='out_relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Create model.
    sModel = tf.keras.Model(img_input, x)

    return sModel


def _inverted_res_block(inputs, in_filters, expansion, stride, filters, block_id):
    pointwise_filters = filters
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = tf.keras.layers.Conv2D(expansion * in_filters,
                                   kernel_size=1,
                                   padding='same',
                                   use_bias=False,
                                   activation=None,
                                   name=prefix + 'expand')(x)
        x = tf.keras.layers.BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
        x = tf.keras.layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = tf.keras.layers.ZeroPadding2D(padding=1,
                                          name=prefix + 'pad')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                        strides=stride,
                                        activation=None,
                                        use_bias=False,
                                        padding='same' if stride == 1 else 'valid',
                                        name=prefix + 'depthwise')(x)
    x = tf.keras.layers.BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = tf.keras.layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = tf.keras.layers.Conv2D(pointwise_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               activation=None,
                               name=prefix + 'project')(x)

    if in_filters == pointwise_filters and stride == 1:
        return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])
    # Might want to figure out something for residual on stride = 2
    return x

# Add the code for upsample here


def upsample(filters, size, apply_dropout=False):
  initializer = tf.keras.initializers.RandomNormal(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


# This is the big question, will this work with a 14d tensor
base_model = SMobileNetV2(input_shape=image_shape)

# tf.keras.utils.plot_model(base_model, to_file='MobileModel.png', show_shapes=True)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 128x32
    'block_3_expand_relu',   # 64x16
    'block_6_expand_relu',   # 32x8
    'block_12_expand_relu',  # 16x4
    'out_relu',      # 8x2
]

# This line is rather complicated but it gets the output from the 1st block
# The 3rd, 6th and so on.
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

up_stack = [
    upsample(512, 3, apply_dropout=True),  # 4x1 -> 8x2
    upsample(512, 3, apply_dropout=True),  # 8x2 -> 16x4
    upsample(256, 3, apply_dropout=True),  # 16x4 -> 32x8
    upsample(128, 3),   # 32x8 -> 64x16
    upsample(64, 3),   # 64x16 -> 128x32
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[256, 64, 15])
    x = inputs

    # Down sampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Up sampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate(axis=3)([x, skip])
        x = concat

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', activation='softmax')

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def my_loss_cat(y_pred, y_true):
    CE = 0
    for c in range(0, 3):
        scale_factor = 1 / (tf.reduce_sum(y_true[:, :, :, c]) + 1)
        # tf.print(y_pred[:, :, :, c])
        CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            tf.math.log(y_pred[:, :, :, c]), tf.float32))) * scale_factor * class_factor[c]
    return CE * -1


def CatCrossEnt(y_true, y_pred):
    CE = 0
    for c in range(0, 3):
        scale_factor = 1 / tf.reduce_sum(y_true[:, :, :, c])
        # tf.print(y_pred[:, :, :, c])
        CE += tf.reduce_sum(tf.multiply(y_true[:, :, :, c], tf.cast(
            tf.math.log(y_pred[:, :, :, c]), tf.float32))) * scale_factor
    return CE * -1


generator = unet_model(OUTPUT_CHANNELS)

# tf.keras.utils.plot_model(generator, to_file='UpsampleModel.png', show_shapes=True)

EPOCHS = 30

log_dir = "logs/"

# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss_object = my_loss_cat
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

generator_optimizer = tf.keras.optimizers.Adam(2e-3, beta_1=0.5)
mobile_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  mobile_optimizer=mobile_optimizer,
#                                  generator=generator,
#                                  mobile=base_model)


def generator_loss(pred, true):
    return loss_object(pred, true)


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as mobile_tape:
    # base_model([input_image], training=True)
    generator_output = generator([input_image], training=True)
    gen_total_loss = generator_loss(generator_output, target)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  mobile_gradients = mobile_tape.gradient(gen_total_loss,
                                          base_model.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  mobile_optimizer.apply_gradients(zip(mobile_gradients,
                                       base_model.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_train_loss', gen_total_loss, step=epoch)
    for t_metric in METRICS:
        # tf.summary.scalar('train_{}'.format(t_metric.name), t_metric(target, generator_output), step=epoch)
        t_metric.update_state(target, generator_output)
        # tf.summary.scalar("{}".format(t_metric.name), t_metric.result(), step=epoch)
        # t_metric.reset_states()


def fit(epochs):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    # if epoch % 5 == 1:
    #   for example_input, example_target in test_data.take(1):
    #     generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    n = 0
    for (input_image, target) in train_data:
      train_step(input_image, target, epoch)
    print()

    with summary_writer.as_default():
        for t_metric in METRICS:
            tf.summary.scalar("train_{}".format(t_metric.name), t_metric.result(), step=epoch)
            t_metric.reset_states()

    for (test_image, test_target) in test_data:
      log_metric(test_image, test_target, epochs, val=True)

    with summary_writer.as_default():
        for t_metric in METRICS:
            tf.summary.scalar("c2_{}".format(t_metric.name), t_metric.result(), step=epoch)
            t_metric.reset_states()

    for (test_image, test_target) in test_data:
      log_metric(test_image, test_target, epochs, val=False)

    with summary_writer.as_default():
        for t_metric in METRICS:
            tf.summary.scalar("val_{}".format(t_metric.name), t_metric.result(), step=epoch)
            t_metric.reset_states()

    for test_image, test_target in test_data.take(5):
        with summary_writer.as_default():
            generator_output = generator(test_image)
            tf.summary.image("gen outputs", generator_output, max_outputs=5, step=epoch)
            tf.summary.image("gen target", test_target, max_outputs=5, step=epoch)

    # saving (checkpoint) the model every 20 epochs
    # if (epoch + 1) % 20 == 0:
    #   checkpoint.save(file_prefix=checkpoint_prefix)

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time()-start))
  # checkpoint.save(file_prefix=checkpoint_prefix)


def log_metric(input_image, target, epoch, val):
    generator_output = generator([input_image])
    gen_test_loss = generator_loss(generator_output, target)
    tar = target.numpy()
    gen_t_out = generator_output.numpy()
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_test_loss', gen_test_loss, step=epoch)
    if ~val:
        for c2_metric in METRICS:
            # tf.summary.scalar('c2_{}'.format(v_metric.name),
            #                   v_metric(tar[:, :, :, -1], gen_t_out[:, :, :, -1]), step=epoch)
            c2_metric.update_state(tar[:, :, :, -1], gen_t_out[:, :, :, -1])
            # tf.summary.scalar("{}".format(v_metric.name), v_metric.result(), step=epoch)
            # v_metric.reset_states()
    else:
        for v_metric in METRICS:
            # tf.summary.scalar('val_{}'.format(metric.name), metric(target, generator_output), step=epoch)
            v_metric.update_state(target, generator_output)
            # tf.summary.scalar("{}".format(metric.name), metric.result(), step=epoch)
            # metric.reset_states()


def generate_images(model, test_input, target):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))
  display_list = [prediction[0], target[0]]
  title = ['Predicted Image', 'Ground Truth']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i][:, :, -1])
    plt.axis('off')
  plt.savefig('/data/TBI/Datasets/Pictures/Evaluator/Mobile/' +
              datetime.datetime.now().strftime("%m%d-%H") + "/" + "fun" + ".png")


fit(EPOCHS)
generator.save('/data/TBI/Datasets/Models/IPH_Mobile_0')

tf.keras.utils.plot_model(generator, to_file='PixModel.png', show_shapes=True)
