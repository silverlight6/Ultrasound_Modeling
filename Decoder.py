import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils import conv_utils


# This may see some changes if I want to use a different style of decoder
class DecoderBlock(tf.Module):
    def __init__(self, out_channels, wDecay=None):
        super().__init__()
        self.wDecay = wDecay
        self.conv1_0 = tf.keras.layers.Conv2D(out_channels // 4, 1, strides=1, padding='SAME',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay)
        self.conv1_1 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay,
                                              dilation_rate=(2, 2))
        self.conv1_2 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay,
                                              dilation_rate=(4, 4))
        self.conv1_3 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay,
                                              dilation_rate=(8, 8))
        self.pool = tf.keras.layers.AveragePooling2D()
        self.conv1_4 = tf.keras.layers.Conv2D(out_channels // 4, 1, strides=1, padding='Same',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay)
        # self.upsample = tf.keras.layers.UpSampling2D()
        self.LeakyReLU1 = tf.keras.layers.LeakyReLU()
        self.bn1_0 = tf.keras.layers.BatchNormalization()
        self.bn1_1 = tf.keras.layers.BatchNormalization()
        self.bn1_2 = tf.keras.layers.BatchNormalization()
        self.bn1_3 = tf.keras.layers.BatchNormalization()
        self.conv2_0 = tf.keras.layers.Conv2D(out_channels // 4, 1, strides=1, padding='SAME',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay)
        self.conv2_1 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay,
                                              dilation_rate=(2, 2))
        self.conv2_2 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay,
                                              dilation_rate=(4, 4))
        self.conv2_3 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              kernel_regularizer=self.wDecay,
                                              dilation_rate=(8, 8))
        self.bn2_0 = tf.keras.layers.BatchNormalization()
        self.bn2_1 = tf.keras.layers.BatchNormalization()
        self.bn2_2 = tf.keras.layers.BatchNormalization()
        self.bn2_3 = tf.keras.layers.BatchNormalization()

        # self.up = tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=4, strides=2, padding='same')
        self.up = tf.keras.layers.Conv2DTranspose(out_channels, 3, strides=2, padding='Same',
                                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                                  kernel_regularizer=wDecay)

    def forward(self, x, skip=None):
        # input comes in [256, 128, 64]
        x = self.up(x)
        if skip is not None:
            # skips come in d = [256, 128, 64]
            x = tf.concat([x, skip], axis=3)
        x1 = self.conv1_0(x)
        x1 = self.bn1_0(x1)
        x2 = self.conv1_1(x)
        x2 = self.bn1_1(x2)
        x3 = self.conv1_2(x)
        x3 = self.bn1_2(x3)
        x4 = self.conv1_3(x)
        x4 = self.bn1_3(x4)
        x = tf.concat([x1, x2, x3, x4], axis=3)
        x = self.LeakyReLU1(x)

        # possible experiment is to reuse the earlier convolutions here.
        x1 = self.conv2_0(x)
        x1 = self.bn2_0(x1)
        x2 = self.conv2_1(x)
        x2 = self.bn2_1(x2)
        x3 = self.conv2_2(x)
        x3 = self.bn2_2(x3)
        x4 = self.conv2_3(x)
        x4 = self.bn2_3(x4)
        x = tf.concat([x1, x2, x3, x4], axis=3)
        x = self.LeakyReLU1(x)

        # output has [128, 64, 16] channels
        return x

    def __call__(self, x, skip=None, *args, **kwargs):
        return self.forward(x, skip)


# This may see some changes if I want to use a different style of decoder
class DecoderCup(tf.Module):
    def __init__(self, num_classes, wDecay=None):
        super().__init__()
        head_channels = 256
        self.wDecay = wDecay
        self.conv_more = tf.keras.layers.Conv2D(
            filters=head_channels,
            kernel_size=3,
            strides=1,
            padding='SAME',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=self.wDecay
        )
        self.LeakyReLU1 = tf.keras.layers.LeakyReLU()
        self.bn1 = tf.keras.layers.LayerNormalization()

        skip_channels = [256, 128, 64]
        blocks = [
            DecoderBlock(sk_ch, wDecay) for sk_ch in skip_channels
        ]
        self.num_classes = num_classes
        self.blocks = blocks
        self.head = tf.keras.layers.Conv2DTranspose(self.num_classes, 3, strides=2, padding='Same',
                                                    activation='softmax', kernel_regularizer=wDecay,
                                                    kernel_initializer=tf.keras.initializers.HeNormal())

    def forward(self, hidden_states, features=None):
        # This line is in shape batch_size, num_patches, hidden_state size
        tfShape = tf.shape(hidden_states)
        y = hidden_states
        x = tf.reshape(tensor=hidden_states, shape=[tfShape[0], 16, 5, -1])
        x = self.conv_more(x)
        x = self.bn1(x)
        x = self.LeakyReLU1(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < 3) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            # print(tfShape)
            # print([tfShape[0], tfShape[1] * tf.pow(2, i + 1), tfShape[2] * tf.pow(2, i + 1), -1])
            x0 = tf.reshape(y, [tfShape[0], 16 * tf.pow(2, i+1), 5 * tf.pow(2, i+1), -1])
            x = tf.concat([x, x0], axis=3)
        x = self.head(x)
        return x

    def __call__(self, hidden_states, features=None, *args, **kwargs):
        return self.forward(hidden_states, features)


# This may see some changes if I want to use a different style of decoder
class KSACBlock(tf.Module):
    def __init__(self, out_channels, wDecay=None):
        super().__init__()
        self.wDecay = wDecay
        self.kernel_size = [3, 3]
        self.out_channels = out_channels
        self.conv1 = KernelSharingConv(self.out_channels, self.kernel_size, kernel_regularizer=self.wDecay,
                                       name='KSAC_1', kernel_initializer='HeNormal')
        self.conv2 = KernelSharingConv(self.out_channels, self.kernel_size, kernel_regularizer=self.wDecay,
                                       name='KSAC_1', kernel_initializer='HeNormal')
        self.up = tf.keras.layers.Conv2DTranspose(out_channels, 3, strides=2, padding='Same', kernel_regularizer=wDecay,
                                                  kernel_initializer=tf.keras.initializers.HeNormal())

    def forward(self, x, skip=None):
        # input comes in [256, 128, 64]
        x = self.up(x)
        if skip is not None:
            # skips come in d = [256, 128, 64]
            x = tf.concat([x, skip], axis=3)
        x = self.conv1(x)
        x = tf.convert_to_tensor(x)
        x = self.conv2(x)
        x = tf.convert_to_tensor(x)
        return x

    def __call__(self, x, skip=None, *args, **kwargs):
        return self.forward(x, skip)


@tf.function
def compute_shift_and_paddings(k_height, k_width, dilation_v=1, dilation_h=1):
    kernel_height_center = k_height // 2
    kernel_width_center = k_width // 2

    shifts_list = []
    paddings_list = []

    for i in range(0, k_height):
        for j in range(0, k_width):
            v_shift = (kernel_height_center - i) * dilation_v
            h_shift = (kernel_width_center - j) * dilation_h

            v_slice_start = -v_shift if v_shift < 0 else 0
            h_slice_start = -h_shift if h_shift < 0 else 0

            v_slice_end = -v_shift if v_shift > 0 else 0
            h_slice_end = -h_shift if h_shift > 0 else 0

            shifts = [v_slice_start, v_slice_end, h_slice_start, h_slice_end]
            shifts_list += [shifts]

            paddings = [[0, 0]]
            paddings += [[v_shift, 0]] if v_shift > 0 else [[0, -v_shift]]
            paddings += [[h_shift, 0]] if h_shift > 0 else [[0, -h_shift]]
            # This is in the original code but like.. wtf..
            paddings += [[0, 0]]

            paddings_list += [paddings]

    return shifts_list, paddings_list


@tf.function
def get_shift_and_paddings(k_height, k_width, dilations=None):
    if dilations is None:
        dilations = [1, 1]
    elif isinstance(dilations, int):
        dilations = [dilations, dilations]

    dilations = list(dilations)

    shifts_list, paddings_list = compute_shift_and_paddings(k_height, k_width, dilations[0], dilations[1])

    return tf.convert_to_tensor(shifts_list), tf.convert_to_tensor(paddings_list)


@tf.function
def kernel_sharing_conv2d(inputs, filters, strides=(1, 1), dilation_rates_list=None, name=None):
    if name is None:
        name = "kernel_sharing_conv2d"

    with tf.name_scope(name=name):

        x = inputs  # [N, H, W, c]
        k = filters  # [h, w, c, C]

        inputs_shape = tf.shape(x)
        batch_size = inputs_shape[0]
        height = inputs_shape[1]
        width = inputs_shape[2]
        in_channels = x.shape[-1]

        k_height, k_width, _, out_channels = k.shape

        k_size = k_height * k_width

        x = tf.raw_ops.Reshape(tensor=x, shape=[batch_size, height * width, in_channels])  # [N, HW, c]
        k = tf.raw_ops.Reshape(tensor=k, shape=[k_size, 1, in_channels, out_channels])

        y = tf.zeros(shape=[batch_size, height, width, out_channels], dtype=inputs.dtype)
        y = y[:, ::strides[0], ::strides[1], :]

        y_shape = tf.TensorShape(y.shape)

        shifts_list_groups = []
        paddings_list_groups = []
        y_groups = []

        for dilations in dilation_rates_list:
            shifts_list, paddings_list = get_shift_and_paddings(k_height, k_width, dilations=dilations)

            shifts_list_groups += [shifts_list]
            paddings_list_groups += [paddings_list]

            y_groups += [y]

        for i in range(k_size):

            value = tf.raw_ops.BatchMatMulV2(x=x, y=k[i])  # [N, HW, C]
            value = tf.raw_ops.Reshape(tensor=value, shape=[batch_size, height, width, out_channels])

            for j in range(len(dilation_rates_list)):
                shifts_list = shifts_list_groups[j]
                paddings_list = paddings_list_groups[j]

                shifts = shifts_list[i]

                v_bottom_shift = tf.raw_ops.SelectV2(condition=shifts[1] == 0, t=height, e=shifts[1])
                h_right_shift = tf.raw_ops.SelectV2(condition=shifts[3] == 0, t=width, e=shifts[3])

                value = value[:, shifts[0]: v_bottom_shift, shifts[2]: h_right_shift, :]  # [N, H - ?, W - ?, C]

                paddings = paddings_list[i]

                value = tf.raw_ops.Pad(input=value, paddings=paddings)
                value = value[:, ::strides[0], ::strides[1], :]
                value = tf.raw_ops.EnsureShape(input=value, shape=y_shape)

                y_groups[j] = tf.raw_ops.Add(x=y_groups[j], y=value)

        return y_groups


# default was (1, 6, 12, 18, 24), switching to (1, 2, 4, 8, 16) for my purposes because of smaller input
class KernelSharingConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rates_list=(1, 2, 4, 8, 16), trainable=True,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None, use_bn=True, name=None):
        super().__init__(trainable=trainable, name=name)

        # Initialized for standard conv, if 3d conv, I believe this should be 3.
        rank = 2

        if isinstance(filters, float):
            filters = int(filters)

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.dilation_rates_list = dilation_rates_list

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.trainable = trainable

        self.use_bn = use_bn

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = input_shape[-1]

        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      dtype=self.dtype)

        if self.use_bn:
            # self.bns = [tf.keras.layers.experimental.SyncBatchNormalization(name=f"bn_r_{r}")
            #             for r in self.dilation_rates_list]
            self.bns = [tf.keras.layers.BatchNormalization(name=f"bn_r_{r}") for r in self.dilation_rates_list]

    def call(self, inputs, training=None):
        y_list = kernel_sharing_conv2d(inputs, filters=self.kernel, dilation_rates_list=self.dilation_rates_list)

        for i in range(len(y_list)):

            x = y_list[i]

            if self.use_bn:
                x = self.bns[i](x, training=training)

            # originally relu but I am switching this to gelu to keep it consistent with the rest of the model.
            y_list[i] = tf.nn.gelu(x)
        return y_list
