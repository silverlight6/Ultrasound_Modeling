import random
import numpy as np
import tensorflow as tf


def shift(image, label):
    r = random.randint(0, 30)
    c = random.randint(0, 12)
    direction = random.randint(0, 1)
    si = image.shape
    mask2 = np.zeros((si[0], si[1], si[2]), dtype=np.float64)
    mask3 = np.zeros((si[0], si[1]), dtype=np.float64)
    for i in range(0, si[0] - 1):
        for j in range(0, si[1] - 1):
            if direction:
                if 0 <= i + r < si[0] and 0 <= j + c < si[1]:
                    mask2[i, j, :] = image[i + r, j + c, :]
                    mask3[i, j] = label[i + r, j + c]
            else:
                if 0 <= i - r < si[0] and 0 <= j - c < si[1]:
                    mask2[i, j, :] = image[i - r, j - c, :]
                    mask3[i, j] = label[i - r, j - c]
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
    row, col, ch = image.shape
    mean = 0
    var = 1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape([row, col, ch])
    gauss /= 5000
    noisyI = image + gauss
    return noisyI


def imageReduc(image, t):
    output = image
    si = image.shape
    # Where it is outside the brain, mark 1
    mask = np.where(image[:, :, 0] < 0.1, 1, 0)
    # Initialize to 0s
    mask2 = np.zeros((si[0], si[1]), dtype=np.int32)
    for _ in range(0, t):                   # shrinks input by 2 pixels
        for i in range(1, si[0] - 1):
            for j in range(1, si[1] - 1):
                if mask[i, j] > 1:
                    mask2[i - 1, j] = 1
                    mask2[i, j - 1] = 1
                    mask2[i + 1, j] = 1
                    mask2[i, j + 1] = 1
                    mask2[i - 1, j - 1] = 1
                    mask2[i - 1, j + 1] = 1
                    mask2[i + 1, j + 1] = 1
                    mask2[i + 1, j - 1] = 1
                    mask2[i, j] = 1
        mask = mask2
        mask2 = np.zeros((si[0], si[1]), dtype=np.int32)
    # shrink the mask by 2
    output[:, :, 0] = np.where(mask == 1, 0, output[:, :, 0])
    for k in range(1, si[2]):
        output[:, :, k] = np.where(output[:, :, 0] == 0, 0, output[:, :, k])

    return output[:, :, 0], output[:, :, 1:]


def dataAug(image, label):
    r = random.randint(0, 100000)
    t = random.randint(0, 100000)

    # flip horizontal
    # if r % 2:
    #     image = np.fliplr(image)
    #     label = np.fliplr(label)

    if r % 3 != 0:
        label, image = imageReduc(np.concatenate([np.expand_dims(label, axis=-1), image], axis=2), t % 7 + 2)

    for _ in range(r % 3):
        label, image = clip(image, label)

    if t % 2:
        label, image = shift(image, label)

    if t % 3:
        image = noisy(image)
    return image, label
