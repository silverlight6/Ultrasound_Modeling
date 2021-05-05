import tensorflow as tf
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt
from datetime import datetime

training_data_path = '/DATA/TBI/Datasets/NPFiles/CardiacBalanced/TrainingData.npy'
testing_data_path = '/DATA/TBI/Datasets/NPFiles/CardiacBalanced/ValidationData.npy'
train_data = np.load(training_data_path)
test_data = np.load(testing_data_path)

# channel 0: outside the brain
# channel 1: no-bleed
# channel 2: bleed
OUTPUT_CHANNELS = 14
BATCH_SIZE = 64
BUFFER_SIZE = 100
xdim = 256
ydim = 64


def preProcess(input_data):
    t_y = tf.gather(input_data, 0, axis=4)
    t_x = tf.gather(input_data, list(range(1, 15)), axis=4)  # weeding out the x data
    t_x = tf.cast(t_x, tf.float64)
    return t_x, t_y  # return input and output


train_data = preProcess(train_data)
test_data = preProcess(test_data)

# load the numpy arrays into TensorFlow dataset object
train_data = tf.data.Dataset.from_tensor_slices(train_data)
test_data = tf.data.Dataset.from_tensor_slices(test_data)

train_data.shuffle(BUFFER_SIZE)  # shuffle the data
train_data.batch(BATCH_SIZE)  # make data into batches, based on batch size

test_data.batch(BATCH_SIZE)  # make data into batches, based on batch size

print(train_data)
print(test_data)
image_shape = [xdim, ydim, 14]


def downsample(filters, size, conv_id, stride=2):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()  # construct Sequential model
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                               kernel_initializer=initializer, use_bias=False,
                               name='conv_{}'.format(conv_id)))

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size):
    initializer = tf.keras.initializers.RandomNormal(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.ReLU())

    return result


class VAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        down_stack = [
            downsample(128, 4, conv_id=1),  # (bs, 128, 32, 64)
            downsample(128, 4, conv_id=1),  # (bs, 64, 16, 128)
            downsample(256, 4, conv_id=2),  # (bs, 32, 8, 256)
            downsample(256, 4, conv_id=3),  # (bs, 16, 4, 256)
            downsample(256, 4, conv_id=4),  # (bs, 8, 2, 256)
            tf.keras.layers.Flatten(),  # (bs, 4096, 1)
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim, dtype=tf.float32),
        ]

        up_stack = [
            tf.keras.layers.Dense(units=8 * 2 * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(8, 2, 32)),
            upsample(256, 4),  # (bs, 16, 4, 256)
            upsample(256, 4),  # (bs, 32, 8, 256)
            upsample(256, 4),  # (bs, 64, 16, 256)
            upsample(128, 4),  # (bs, 128, 32, 128)
            upsample(OUTPUT_CHANNELS, 4),  # (bs, 256, 64, 14)
        ]

        inputs_encode = tf.keras.layers.Input(shape=[xdim, ydim, 14])
        x = inputs_encode
        # Downsampling through the model
        # iterate over the downsample layers and connect them together
        for down in down_stack:
            x = down(x)

        self.encoder = tf.keras.Model(inputs=inputs_encode, outputs=x)

        inputs_decode = tf.keras.layers.Input(shape=(latent_dim,))
        y = inputs_decode
        # Upsampling and establishing the skip connections

        for up in up_stack:
            y = up(y)

        self.decoder = tf.keras.Model(inputs=inputs_decode, outputs=y)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        output = self.encoder(x)
        mean, logvar = tf.split(output, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mu, sigma):
        eps = tf.random.normal(tf.shape(mu), 0, 1)
        sigma = tf.cast(sigma, tf.float32)
        mu = tf.cast(mu, tf.float32)
        return eps * tf.exp(0.5 * sigma) + mu

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


beta = 4
optimizer = tf.keras.optimizers.Adam(2e-3)
total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
mse_tracker = tf.keras.metrics.MeanSquaredError(name='mse_loss')


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * beta * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mu, sigma = model.encode(x)

    z = model.reparameterize(mu, sigma)
    x_logit = model.decode(z)
    mse_tracker.update_state(x, x_logit)

    # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    # logpz = log_normal_pdf(z, 0., 0.)
    # logqz_x = log_normal_pdf(z, mu, sigma)
    # total_loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
    #
    # return total_loss, logpx_z, logpz - logqz_x
    # Try 2 for another tutorial
    marginal_likelihood = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_logit), axis=(0, 1, 2))
    KL_divergence = tf.reduce_sum(-0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma)), 1)
    return tf.reduce_mean(marginal_likelihood) + beta * tf.reduce_mean(KL_divergence), marginal_likelihood, KL_divergence


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
    with tf.GradientTape() as tape:
        loss, recon_loss, kl_loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    total_loss_tracker.update_state(loss)
    reconstruction_loss_tracker.update_state(recon_loss)
    kl_loss_tracker.update_state(kl_loss)


epochs = 250
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 512

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
model = VAE(latent_dim)
# model.build()


def generate_images(model, epoch, test_sample, label):
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=.25, wspace=.3, bottom=.1)
    fig.set_size_inches(10, 6)
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    test_sample = tf.reshape(test_sample, [256, 64, 14])
    predictions = tf.reshape(predictions, [256, 64, 14])
    difference = test_sample - predictions
    kernel = np.ones((5, 5), np.float32) / 25
    difference = cv2.filter2D(difference, -1, kernel)
    difference = np.abs(difference)

    for i in range(0, 14):
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[0, 0].grid(False)
        ax[0, 0].title.set_text('Prediction')
        ax[0, 0].imshow(predictions[:, :, i], cmap='autumn')
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].grid(False)
        ax[0, 1].title.set_text('Label at epoch {}'.format(epoch))
        ax[0, 1].imshow(test_sample[:, :, i], cmap='autumn')
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        ax[1, 0].grid(False)
        ax[1, 0].title.set_text('Difference')
        ax[1, 0].imshow(difference[:, :, i], cmap='autumn')
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].grid(False)
        ax[1, 1].title.set_text('Label')
        ax[1, 1].imshow(y, cmap='autumn')
        plt.show()


for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x, _ in train_data:
        train_step(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x, y in test_data.take(2):
        loss, _, _ = compute_loss(model, test_x)
        if epoch % 25 == 0:
            generate_images(model, epoch, test_x, y)
    for train_x, y in train_data.take(2):
        loss, _, _ = compute_loss(model, train_x)
        if epoch % 25 == 0:
            generate_images(model, epoch, train_x, y)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, loss, end_time - start_time))
    print('Total loss: {}, Reconstruction loss: {}, KL Divergence: {}, MSE: {}'
          .format(total_loss_tracker.result(), reconstruction_loss_tracker.result(),
                  kl_loss_tracker.result(), mse_tracker.result()))

model.save('TBI_AutoEncoder')
