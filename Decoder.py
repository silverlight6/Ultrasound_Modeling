import tensorflow as tf


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
        self.bn1 = tf.keras.layers.BatchNormalization()

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
