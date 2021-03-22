import tensorflow as tf
import numpy as np
import os
import time
import datetime

from IPython import display
from matplotlib import pyplot as plt

training_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/TrainingPolarData.npy'
testing_data_path = '/home/silver/Documents/TBI_NNs/Datasets/NPFiles/TestingPolarData.npy'

OUTPUT_CHANNELS = 3
BATCH_SIZE = 1
BUFFER_SIZE = 100


def preProcess(input_data):
    t_y = tf.gather(input_data, 0, axis=3)
    t_x = tf.gather(input_data, list(range(1, 15)), axis=3)
    t_y = tf.cast(t_y, dtype=tf.int32)
    # t_y = tf.reshape(t_y, [1, 160, 192, 1])
    print(t_y)
    t_y = tf.one_hot(t_y, depth=OUTPUT_CHANNELS)
    print(t_y)
    return t_x, t_y


train_data = tf.data.Dataset.from_tensor_slices(np.load(training_data_path))
test_data = tf.data.Dataset.from_tensor_slices(np.load(testing_data_path))
train_data = train_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_data.shuffle(BUFFER_SIZE)
train_data.batch(BATCH_SIZE)

test_data.batch(BATCH_SIZE)


image_shape = [256, 64, 14]


def downsample(filters, size, apply_batchnorm=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

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


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 64, 14])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (bs, 96, 80, 64)
    downsample(256, 4),  # (bs, 48, 40, 256)
    downsample(512, 4),  # (bs, 24, 20, 512)
    downsample(512, 4),  # (bs, 12, 10, 512)
    downsample(512, 4),  # (bs, 6, 5, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (bs, 12, 10, 1024)
    upsample(512, 4, apply_dropout=True),  # (bs, 24, 20, 1024)
    upsample(256, 4),  # (bs, 48, 40, 768)
    upsample(128, 4),  # (bs, 96, 80, 384)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='softmax')  # (bs, 192, 160, 4)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 64, 14], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 64, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (bs, 192, 160, channels*2)

  down1 = downsample(64, 4, False)(x)  # (bs, 96, 80, 64)
  down2 = downsample(128, 4)(down1)  # (bs, 48, 40, 128)
  down3 = downsample(256, 4)(down2)  # (bs, 24, 20, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 26, 22, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 25, 21, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 27, 23, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 24, 20, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_images(model, test_input, target):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [target[0], prediction[0]]
  title = ['Ground Truth', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(tf.cast(tf.argmax(display_list[i], axis=-1), tf.float64) * 0.25)
    plt.axis('off')
  plt.show()


EPOCHS = 40

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(epochs):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    if epoch % 10 == 0:
      for example_input, example_target in test_data.take(1):
        generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    n = 0
    for (input_image, target) in train_data:
      print('.', end='')
      n = n + 1
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time()-start))
  checkpoint.save(file_prefix=checkpoint_prefix)


fit(EPOCHS)

display.IFrame(
    src="https://tensorboard.dev/experiment/lZ0C6FONROaUMfjYkVyJqw",
    width="100%",
    height="1000px")

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on a few examples from the test dataset
for k, j in test_data.take(5):
  generate_images(generator, k, j)
