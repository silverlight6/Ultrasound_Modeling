import random
import numpy as np
import tensorflow as tf


def shift(image, label):
    r = random.randint(0, 30)
    c = random.randint(0, 12)
    direction = random.randint(0, 1)
    si = image.shape
    mask2 = np.zeros((si[0], si[1], si[2], si[3]), dtype=np.float64)
    mask3 = np.zeros((si[0], si[1], si[2], 1), dtype=np.float64)
    for i in range(0, si[1] - 1):
        for j in range(0, si[2] - 1):
            if direction:
                if 0 <= i + r < si[1] and 0 <= j + c < si[2]:
                    mask2[:, i, j, :] = image[:, i + r, j + c, :]
                    mask3[:, i, j, :] = label[:, i + r, j + c, :]
            else:
                if 0 <= i - r < si[1] and 0 <= j - c < si[2]:
                    mask2[:, i, j, :] = image[:, i - r, j - c, :]
                    mask3[:, i, j, :] = label[:, i - r, j - c, :]
    return mask3, mask2


def clip(image, label):
    r = random.randint(0, 256)
    c = random.randint(0, 80)
    ra = random.randint(20, 40)
    ca = random.randint(10, 20)
    si = image.shape
    for i in range(0, si[0] - 1):
        for j in range(0, si[1] - 1):
            if r + ra > i > r - ra and c + ca > j > c - ca:
                image[i, j, :] = 0
                label[i, j] = 0
    return label, image


def noisy(image):
    batch, row, col, ch = image.shape
    mean = 0
    var = 1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (batch, row, col, ch))
    gauss = gauss.reshape([batch, row, col, ch])
    gauss /= 5000
    noisyI = image + gauss
    return noisyI


def imageReduc(label, image, t):
    output = image
    si = image.shape
    # Where it is outside the brain, mark 1
    mask = np.where(label[:, :, :, :] < 0.1, 1, 0)
    # Initialize to 0s
    mask2 = np.zeros((si[0], si[1], si[2]), dtype=np.int32)
    for _ in range(0, t):                   # shrinks input by 2 pixels
        for i in range(1, si[0] - 1):
            for j in range(1, si[1] - 1):
                if mask[:, i, j] > 1:
                    mask2[:, i - 1, j] = 1
                    mask2[:, i, j - 1] = 1
                    mask2[:, i + 1, j] = 1
                    mask2[:, i, j + 1] = 1
                    mask2[:, i - 1, j - 1] = 1
                    mask2[:, i - 1, j + 1] = 1
                    mask2[:, i + 1, j + 1] = 1
                    mask2[:, i + 1, j - 1] = 1
                    mask2[:, i, j] = 1
        mask = mask2
        mask2 = np.zeros((si[0], si[1], si[2]), dtype=np.int32)
    # shrink the mask by 2
    # print("mask.shape = {}".format(mask.shape))
    label[:, :, :, 0] = np.where(mask == 1, 0, label[:, :, :, 0])
    for k in range(0, si[3]):
        output[:, :, :, k] = np.where(label[:, :, :, 0] == 0, 0, output[:, :, :, k])
    return label, output


def dataAug(image, label):
    r = random.randint(0, 100000)
    t = random.randint(0, 100000)

    image = image.numpy()
    label = label.numpy()

    # flip horizontal
    if r % 2:
        image = np.fliplr(image)
        label = np.fliplr(label)

    if t % 2:
        label, image = imageReduc(label, image, r % 3 + 1)

    # # Contrast
    # if t % 3:
    #     # With norm, use this
    #     image = image - np.min(image, keepdims=True) / (np.max(image, keepdims=True) - np.min(image, keepdims=True))
    #     minval = np.percentile(image, 2)
    #     maxval = np.percentile(image, 98)
    #     image = np.clip(image, minval, maxval)
    #     # if norm use this
    #     image = (image - minval) / ((maxval - minval) + 1e-6)
    #     for i in range(0, image.shape[-1]):
    #         image[:, :, i] = np.where(label < 0.1, 0.0, image[:, :, i])

    if r % 3:
        label, image = clip(image, label)

    # Rotate is not realistic for our application plus it tends to look poor.
    # if r % 7:
    #     image = ndimage.rotate(image, (r % 11) / 5, reshape=False)
    #     image = np.array(image)
    #     label = ndimage.rotate(label, (r % 11) / 5, reshape=False)
    #     label = np.array(label)

    if t % 3:
        label, image = shift(image, label)

    if t % 5:
        image = noisy(image)

    # print("image output shape = {}".format(image.shape))
    # print("label output shape = {}".format(label.shape))
    image = tf.convert_to_tensor(image, dtype=tf.float64)
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    return image, label
