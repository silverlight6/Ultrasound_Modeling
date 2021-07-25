import tensorflow as tf


class ResNest(tf.Module):
    # I'm still having issues figuring out what I can put in the init and what I can't.
    # Most of this is self-explanatory. The class_factor and alpha are optional parameters in the loss function
    def __init__(self, height, width, channel, ksize, radix=4, kpaths=4, wDecay=None):
        # print("\nInitializing Short-ResNeSt...")
        super(ResNest, self).__init__()
        self.height, self.width, self.channel = height, width, channel
        self.ksize = ksize
        self.radix, self.kpaths = radix, kpaths
        self.wDecay = wDecay
        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=1, padding='SAME', kernel_regularizer=self.wDecay,
                                            kernel_initializer=tf.keras.initializers.HeNormal(), name="initial_conv")
        self.conv1_act = tf.keras.layers.LeakyReLU()
        self.convtmp_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', kernel_regularizer=self.wDecay,
                                                kernel_initializer=tf.keras.initializers.HeNormal(),)
        self.convtmp_1bn = tf.keras.layers.experimental.SyncBatchNormalization()
        self.convtmp_1act = tf.keras.layers.LeakyReLU()
        self.convtmp_2 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', kernel_regularizer=self.wDecay,
                                                kernel_initializer=tf.keras.initializers.HeNormal(),)
        self.convtmp_2bn = tf.keras.layers.experimental.SyncBatchNormalization()
        self.convtmp_2act = tf.keras.layers.LeakyReLU()
        self.conv1_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv2_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv3_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv4_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv_1 = residual_S(ksize=self.ksize, outchannel=64, radix=self.radix,
                                 kpaths=self.kpaths, wDecay=self.wDecay)
        self.conv_2 = residual_S(ksize=self.ksize, outchannel=128, radix=self.radix,
                                 kpaths=self.kpaths, wDecay=self.wDecay)
        self.conv_3 = residual_S(ksize=self.ksize, outchannel=256, radix=self.radix,
                                 kpaths=self.kpaths, wDecay=self.wDecay)
        self.conv_4 = residual_S(ksize=self.ksize, outchannel=512, radix=self.radix,
                                 kpaths=self.kpaths, wDecay=self.wDecay)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_act(x)
        x = self.convtmp_1(x)
        x = self.convtmp_1bn(x)
        x = self.convtmp_1act(x)
        x = self.convtmp_2(x)
        x = self.convtmp_2bn(x)
        x = self.convtmp_2act(x)
        x = self.conv1_pool(x)
        x_1 = self.conv_1(x)
        x = self.conv2_pool(x_1)
        x_2 = self.conv_2(x)
        x = self.conv3_pool(x_2)
        x_3 = self.conv_3(x)
        x = self.conv4_pool(x_3)
        x_4 = self.conv_4(x)
        return x_4, [x_3, x_2, x_1]

    def __call__(self, x, *args, **kwargs):
        return self.forward(x)


class residual_S(tf.Module):
    def __init__(self, ksize, outchannel, radix, kpaths, atrous=1, wDecay=None):
        super(residual_S, self).__init__()
        self.kpaths = kpaths
        self.atrous = atrous
        self.ksize = ksize
        self.outchannel = outchannel
        self.radix = radix
        self.wDecay = wDecay

        self.cardinal_blocks = []
        for _ in range(self.kpaths):
            cardinal_block = cardinal(self.ksize, self.outchannel // 2, self.radix,
                                      self.kpaths, atrous=self.atrous, wDecay=self.wDecay)
            self.cardinal_blocks.append(cardinal_block)

        self.concats_2 = tf.keras.layers.Conv2D(self.outchannel, self.ksize, strides=1, padding='SAME',
                                                dilation_rate=(self.atrous, self.atrous),
                                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                                kernel_regularizer=self.wDecay)

        self.convtmp_sc = tf.keras.layers.Conv2D(self.outchannel, 1, strides=1, padding='SAME',
                                                 dilation_rate=(self.atrous, self.atrous),
                                                 kernel_initializer=tf.keras.initializers.HeNormal(),
                                                 kernel_regularizer=self.wDecay)
        self.convtmp_scbn = tf.keras.layers.LayerNormalization()
        self.convtmp_scact = tf.keras.layers.LeakyReLU()

    def forward(self, x):
        concats_1 = None
        for layer_block in self.cardinal_blocks:
            cardinalI = layer_block(x)
            if concats_1 is None:
                concats_1 = cardinalI
            else:
                concats_1 = tf.concat([concats_1, cardinalI], axis=3)

        concats_2 = self.concats_2(concats_1)
        x = self.convtmp_sc(x)
        x = self.convtmp_scbn(x)
        x = self.convtmp_scact(x)
        output = x + concats_2

        return output

    def __call__(self, x, *args, **kwargs):
        return self.forward(x)


class cardinal(tf.Module):
    def __init__(self, ksize, outchannel, radix, kpaths, atrous=1, wDecay=None):
        super(cardinal, self).__init__()
        self.outchannel = outchannel
        self.ksize = ksize
        self.radix = radix
        self.kpaths = kpaths
        self.atrous = atrous
        self.wDecay = wDecay

        outchannel_cv11 = int(self.outchannel / self.radix / self.kpaths)
        outchannel_cvkk = int(self.outchannel / self.kpaths)
        self.conv1 = tf.keras.layers.Conv2D(outchannel_cv11, 1, strides=1, padding='SAME',
                                            dilation_rate=(self.atrous, self.atrous), kernel_regularizer=self.wDecay,
                                            kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv1_bn = tf.keras.layers.LayerNormalization()
        self.conv1_act = tf.keras.layers.LeakyReLU()

        self.conv2 = tf.keras.layers.Conv2D(outchannel_cvkk, self.ksize,
                                            strides=1, padding='SAME', dilation_rate=(self.atrous, self.atrous),
                                            kernel_initializer=tf.keras.initializers.HeNormal(),
                                            kernel_regularizer=self.wDecay)
        self.conv2_bn = tf.keras.layers.LayerNormalization()
        self.conv2_act = tf.keras.layers.LeakyReLU()
        self.split = split_attention(outchannel_cvkk, self.radix, self.atrous, self.wDecay)

    def forward(self, x):
        inputs = []
        for idx_r in range(self.radix):
            y = self.conv1(x)
            y = self.conv1_bn(y)
            y = self.conv1_act(y)
            y = self.conv2(y)
            y = self.conv2_bn(y)
            y = self.conv2_act(y)
            inputs.append(y)

        return self.split(inputs)

    def __call__(self, x, *args, **kwargs):
        return self.forward(x)


class split_attention(tf.Module):
    def __init__(self, inchannel, radix, atrous=1, wDecay=None):
        super(split_attention, self).__init__()
        self.inchannel = inchannel
        self.atrous = atrous
        self.radix = radix
        self.wDecay = wDecay
        self.dense1 = tf.keras.layers.Conv2D(self.inchannel // 2, 1, strides=1, padding='SAME',
                                             dilation_rate=(self.atrous, self.atrous),
                                             kernel_initializer=tf.keras.initializers.HeNormal(),
                                             kernel_regularizer=self.wDecay)
        self.dense1_bn = tf.keras.layers.LayerNormalization()
        self.dense1_act = tf.keras.layers.LeakyReLU()
        self.dense2 = tf.keras.layers.Conv2D(self.inchannel, 1, strides=1, padding='SAME',
                                             dilation_rate=(self.atrous, self.atrous),
                                             kernel_initializer=tf.keras.initializers.HeNormal(),
                                             kernel_regularizer=self.wDecay)

    def forward(self, inputs):
        input_holder = None
        for idx_i, inputi in enumerate(inputs):
            if idx_i == 0:
                input_holder = inputi
            else:
                input_holder += inputi

        ga_pool = tf.math.reduce_mean(input_holder, axis=(1, 2))
        y = tf.expand_dims(tf.expand_dims(ga_pool, axis=1), axis=1)

        y = self.dense1(y)
        y = self.dense1_bn(y)
        y = self.dense1_act(y)

        output_holder = None
        for idx_r in range(self.radix):
            z = self.dense2(y)
            if self.radix == 1:
                z = tf.keras.activations.sigmoid(z)
            elif self.radix > 1:
                z = tf.keras.activations.softmax(z)

            if idx_r == 0:
                output_holder = inputs[idx_r] * z
            else:
                output_holder += inputs[idx_r] * z

        return output_holder

    def __call__(self, inputs, *args, **kwargs):
        return self.forward(inputs)

