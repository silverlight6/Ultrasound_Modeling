# # Code is based on the github --> https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling.py
# # There will be A LOT of changes to it but the general structure is very similar
# # Biggest change is moving everything from pytorch to tensorflow
# # Second biggest change is moving from using a prebuilt ResNet base to a untrained and hand coded ResNeSt base.
# # The third change is I don't do a downsampling when flattening for patches to the transformer like the paper calls for

import datetime
import tensorflow as tf
import numpy as np

wDecay= tf.keras.regularizers.L2(l2=0.00001)
# wDecay = None
tf.executing_eagerly()

class Attention(tf.Module):
    def __init__(self, num_heads=4, attention_head_size=512, attention_dropout_rate=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = attention_head_size
        self.qkv_size = attention_head_size // self.num_heads
        self.query = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=wDecay, dtype=tf.float32)
        self.key = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=wDecay, dtype=tf.float32)
        self.value = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=wDecay, dtype=tf.float32)
        self.out = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=wDecay, dtype=tf.float32)
        self.attn_dropout = tf.keras.layers.Dropout(attention_dropout_rate)
        self.proj_dropout = tf.keras.layers.Dropout(attention_dropout_rate)

        self.softmax = tf.keras.layers.Softmax(axis=3)


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.qkv_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def forward(self, hidden_states):
        batch_size = tf.shape(hidden_states)[0]
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        query_layer = self.split_heads(query_layer, batch_size)
        key_layer = self.split_heads(key_layer, batch_size)
        value_layer = self.split_heads(value_layer, batch_size)

        attention_scores = tf.matmul(a=query_layer, b=key_layer, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.num_heads, dtype=tf.float32))
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        reshaped_layer = tf.reshape(context_layer, (batch_size, -1, self.hidden_size))
        attention_output = self.out(reshaped_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights

    def __call__(self, hidden_states, *args, **kwargs):
        atte, weights = self.forward(hidden_states)
        return atte, weights


class Mlp(tf.Module):
    def __init__(self, hidden_size=512, mlp_dim=2048, dropout_rate=0.0):
        super(Mlp, self).__init__()
        self.fc1 = tf.keras.layers.Dense(mlp_dim, dtype=tf.float32)
        self.fc2 = tf.keras.layers.Dense(hidden_size, dtype=tf.float32)
        # self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = tf.keras.activations.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def __call__(self, x, *args, **kwargs):
        out = self.forward(x)
        return out


class Embeddings(tf.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, img_size, hidden_size=512, dropout_rate=0.0):
        super(Embeddings, self).__init__()
        self.img_size = img_size
        self.hidden_size = hidden_size
        grid_size = (16, 5)
        # patch_size = 8 x 10
        patch_size = (img_size[0] // 8 // grid_size[0], img_size[1] // 8 // grid_size[1])
        # real means what it would correlate to for a full size image or 64 x 80
        patch_size_real = (patch_size[0] * 8, patch_size[1] * 8)
        self.seq_len = grid_size[0] * grid_size[1]
        self.n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        # # This line is going to the ResNest model. Change this line out if using 1 or 3d input and are wanting to
        # # use a transfer learning network.
        # self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1)
        self.hybrid_model = ResNest(256, 80, 10, radix=3, ksize=3, kpaths=3)

        # Try using a reshape instead of a convolution later on.
        # self.patch_embeddings = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, padding='same',
        #                                                strides=patch_size, kernel_regularizer=wDecay)
        self.patch_embeddings = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=1, padding='valid',
                                                       strides=1, kernel_regularizer=wDecay)
        self.position_embeddings = tf.zeros([1, self.seq_len, self.hidden_size])

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)  # (B, hidden. grid[0], grid[1])
        x = tf.reshape(x, [-1, self.seq_len, self.hidden_size])
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

    def __call__(self, x, *args, **kwargs):
        x = self.forward(x)
        return x


class Block(tf.Module):
    def __init__(self, hidden_size=512):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def __call__(self, x, *args, **kwargs):
        x = self.forward(x)
        return x


class Encoder(tf.Module):
    def __init__(self, xdim, ydim, num_layers=4):
        super(Encoder, self).__init__()
        self.xDim = xdim
        self.yDim = ydim
        self.encoder_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        self.Transformer_layers = []
        for _ in range(num_layers):
            Transformer_layers = Block()
            self.Transformer_layers.append(Transformer_layers)


    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.Transformer_layers:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

    def __call__(self, hidden_states, *args, **kwargs):
        hidden_states = self.forward(hidden_states)
        return hidden_states


class Transformer(tf.Module):
    def __init__(self, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = Encoder(img_size[0], img_size[1])

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features

    def __call__(self, input_ids, *args, **kwargs):
        return self.forward(input_ids)


# This may see some changes if I want to use a different style of decoder
class DecoderBlock(tf.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1_0 = tf.keras.layers.Conv2D(out_channels // 4, 1, strides=1,padding='SAME',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         kernel_regularizer=wDecay
        )
        self.conv1_1 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         kernel_regularizer=wDecay,
                                         dilation_rate=(2, 2))
        self.conv1_2 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         kernel_regularizer=wDecay,
                                         dilation_rate=(4, 4))
        self.conv1_3 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         kernel_regularizer=wDecay,
                                         dilation_rate=(8, 8))

        self.LeakyReLU1 = tf.keras.layers.LeakyReLU()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2_0 = tf.keras.layers.Conv2D(out_channels // 4, 1, strides=1, padding='SAME',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         kernel_regularizer=wDecay
                                         )
        self.conv2_1 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         kernel_regularizer=wDecay,
                                         dilation_rate=(2, 2))
        self.conv2_2 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         kernel_regularizer=wDecay,
                                         dilation_rate=(4, 4))
        self.conv2_3 = tf.keras.layers.Conv2D(out_channels // 4, 3, strides=1, padding='SAME',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         kernel_regularizer=wDecay,
                                         dilation_rate=(8, 8))
        self.bn2 = tf.keras.layers.BatchNormalization()
        # self.up = tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=4, strides=2, padding='same')
        self.up = tf.keras.layers.UpSampling2D(size=2)

    def forward(self, x, skip=None):
        # input comes in [256, 128, 64]
        x = self.up(x)
        if skip is not None:
            # skips come in d = [256, 128, 64]
            x = tf.concat([x, skip], axis=3)
        x1 = self.conv1_0(x)
        x2 = self.conv1_1(x)
        x3 = self.conv1_2(x)
        x4 = self.conv1_3(x)
        x = tf.concat([x1, x2, x3, x4], axis=3)
        x = self.bn1(x)
        x = self.LeakyReLU1(x)
        # possible experiment is to reuse the earlier convolutions here.
        x1 = self.conv2_0(x)
        x2 = self.conv2_1(x)
        x3 = self.conv2_2(x)
        x4 = self.conv2_3(x)
        x = tf.concat([x1, x2, x3, x4], axis=3)
        x = self.bn2(x)
        x = self.LeakyReLU1(x)
        # output has [128, 64, 16] channels
        return x

    def __call__(self, x, skip=None, *args, **kwargs):
        return self.forward(x, skip)


# This may see some changes if I want to use a different style of decoder
class DecoderCup(tf.Module):
    def __init__(self, num_classes):
        super().__init__()
        head_channels = 256
        self.conv_more = tf.keras.layers.Conv2D(
            filters=head_channels,
            kernel_size=3,
            strides=1,
            padding='SAME',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=wDecay
        )
        self.LeakyReLU1 = tf.keras.layers.LeakyReLU()
        self.bn1 = tf.keras.layers.BatchNormalization()

        skip_channels = [256, 128, 64]
        blocks = [
            DecoderBlock(sk_ch) for sk_ch in skip_channels
        ]
        self.num_classes = num_classes
        self.blocks = blocks
        self.head = tf.keras.layers.Conv2D(self.num_classes, kernel_size=3, padding='SAME', activation='softmax',
                                           kernel_initializer=tf.initializers.HeNormal, kernel_regularizer=wDecay)
        self.up = tf.keras.layers.UpSampling2D(size=2)

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
        x = self.up(x)
        x = self.head(x)
        return x

    def __call__(self, hidden_states, features=None, *args, **kwargs):
        return self.forward(hidden_states, features)

class VisionTransformer(tf.Module):
    def __init__(self, img_size=(256, 80), num_classes=3, learning_rate=1e-3):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.transformer = Transformer(img_size)
        self.decoder = DecoderCup(num_classes)
        self.initialize = self.forward(np.zeros([16, 256, 80, 10]))
        self.visionModel = self.model()
        # https://github.com/keras-team/keras/blob/master/keras/losses.py
        # I am today years old when I realized keras makes their loss implementations public.
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        # self.loss = self.my_loss_cat
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.alpha = 2
        self.class_factor = [0.06329, 0.027567, 0.90914]


    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        logits = self.decoder(x, features)
        return logits, attn_weights

    '''
    jit_compile simply allows the model to use more of the GPU space available to it.
    This method is the training loop for the model.
    GradientTape records all of the partials for the model and backprops to update weights
    The bottom is a few calculations for metrics purposes.
    '''
    # @tf.function(jit_compile=True)
    def step(self, x, y, train=False):

        with tf.GradientTape() as tape:
            # # Really not sure if this line of code is helpful or just slows things down. Uncomment if you want.
            tape.watch(self.visionModel.trainable_variables)
            logits, _ = self.forward(x)
            smce = self.loss(y_true=y, y_pred=logits)
            smce += sum(self.visionModel.losses)
        if train:
            gradients = tape.gradient(smce, self.visionModel.trainable_variables)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.optimizer.apply_gradients(zip(clip_gradients, self.visionModel.trainable_variables))
        return smce, logits

    def model(self):
        inputA = tf.keras.layers.Input(shape=[256, 80, 10])
        output = self.forward(inputA)
        model = tf.keras.Model(inputs=inputA, outputs=output)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.forward(x)


    @tf.function
    def my_loss_cat(self, y_true, y_pred):
        # CE = 0
        y_true *= 0.9
        y_true += (0.1 / self.num_classes)
        # scale_factor = tf.cast(1 / tf.reduce_sum(y_true, axis=0), tf.float32)
        # scale_factor = tf.divide(x=scale_factor, y=256*80)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)

        # just a fun experiment
        # randOffSet = tf.random.normal(shape=[256, 80], mean=0, stddev=1)
        CE = -3 * tf.reduce_sum(y_true * tf.cast(tf.math.log(y_pred), tf.float32) * self.class_factor, axis=[0,3])

        # CE = -tf.reduce_sum(y_true * tf.cast(tf.math.log(y_pred), tf.float32) * tf.pow(1.0 - y_pred, self.alpha), axis=[0, 3])
        # CE = -3 * tf.reduce_sum(y_true * tf.cast(tf.math.log(y_pred), tf.float32), axis=3)
        # CE += randOffSet * 0.1

        # -3 if using class factor, -1 otherwise. (class factor in my implementation is divides by number of classes)
        # tf.print(CE)
        return CE


class Dataset(object):

    def __init__(self, train_path=None, val_path=None):

        print("\nInitializing Dataset...")
        train_data = np.load(train_path, allow_pickle=True)
        val_data = np.load(val_path, allow_pickle=True)

        # The first 0 is due to how the .append works in the playground file
        # The second 0 is because the label is in the first layer of the data.
        y_tr = train_data[:, 0, :, :, 0]
        y_te = val_data[:, 0, :, :, 0]
        train_data = np.delete(train_data, 0, 4)
        val_data = np.delete(val_data, 0, 4)
        x_tr = np.array(train_data)
        x_te = np.array(val_data)
        # The -1 here is because the last layer is the bMode and I am not using the bMode in the training data
        # This is simply my choice, feel free to change that but be aware that the number of input layers
        # Moves from 10 to 11 and that will affect some lines of code in the evaluator file.
        x_tr = x_tr[:, 0, :, :, :-1]
        x_te = x_te[:, 0, :, :, :-1]
        # This is float64 by default but needs to be float 32 for np.where function.
        y_tr = y_tr.astype(dtype=np.float32)
        y_te = y_te.astype(dtype=np.float32)

        # This is converting the labels from 1d to 3d probability maps
        # 1.05 because resize can sometimes change values up and down by about .01 at any given pixel
        # if > 1 = 1 is because resize sometimes makes values at 101% up to 105%. Putting a cap to ensure
        # this behavior is stopped.
        # The classes don't need to add up to 100. It is simply a nice thing if they do.
        # You only really need each pixel of importance to have a contribution to the loss.
        class_2 = np.where(y_tr >= 1.05, y_tr - 1, 0)
        class_2 = np.where(class_2 > 1, 1, class_2)
        class_1 = np.expand_dims(np.where(y_tr > 0.95, 1 - class_2, 0), axis=3)
        class_0 = np.expand_dims(np.where(y_tr <= 0.95, 1, 0), axis=3)
        class_2 = np.expand_dims(class_2, axis=3)
        y_tr = np.concatenate((class_0, class_1, class_2), axis=3)
        y_tr = tf.convert_to_tensor(y_tr, dtype=tf.float32)
        self.x_tr, self.y_tr = x_tr, y_tr

        class_2 = np.where(y_te >= 1.05, y_te - 1, 0)
        class_2 = np.where(class_2 > 1, 1, class_2)
        class_1 = np.expand_dims(np.where(y_te > 0.95, 1 - class_2, 0), axis=3)
        class_0 = np.expand_dims(np.where(y_te <= 0.95, 1, 0), axis=3)
        class_2 = np.expand_dims(class_2, axis=3)
        y_te = np.concatenate((class_0, class_1, class_2), axis=3)
        y_te = tf.convert_to_tensor(y_te, dtype=tf.float32)
        self.x_te, self.y_te = x_te, y_te

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        self.y_tr = y_tr
        self.y_te = y_te

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        print("Number of data\nTraining: %d, Test: %d\n" % (self.num_tr, self.num_te))
        print("x_tr shape = {}".format(x_tr.shape))
        print("y_tr shape = {}".format(y_tr.shape))

        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        try:
            self.channel = x_sample.shape[2]
        except:
            self.channel = 1

        self.min_val, self.max_val = x_sample.min(), x_sample.max()
        # self.num_class = int(np.floor(y_te.max()+1))
        self.num_class = 3

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" % (self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" % (self.min_val, self.max_val))
        print("Class  %d" % self.num_class)

    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0


    # Get the next batch of training data
    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size
        x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]

        terminator = False
        if end >= self.num_tr:
            terminator = True
            self.idx_tr = 0
            # self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else: self.idx_tr = end

        if fix: self.idx_tr = start

        if x_tr.shape[0] != batch_size:
            x_tr, y_tr = self.x_tr[-1-batch_size:-1], self.y_tr[-1-batch_size:-1]

        return x_tr, y_tr, terminator

    # Get the next batch of test data. This is always 1 for batch size in this model up to this point
    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te + batch_size
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]

        terminator = False
        if end >= self.num_te:
            terminator = True
            self.idx_te = 0
        else: self.idx_te = end

        if x_te.shape[0] != batch_size:
            x_te, y_te = self.x_te[-1-batch_size:-1], self.y_te[-1-batch_size:-1]
        return x_te, y_te, terminator


summary_writer = tf.summary.create_file_writer("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
precision = tf.keras.metrics.Precision(name='precision')
recall = tf.keras.metrics.Recall(name='recall')
pre_c2 = tf.keras.metrics.Precision(name='precision_c2')
re_c2 = tf.keras.metrics.Recall(name='recall_c2')
mio = tf.keras.metrics.MeanIoU(name='mean_iou', num_classes=3)
tr_recall = tf.keras.metrics.Recall(name='tr_recall')
tr_precision = tf.keras.metrics.Precision(name='tr_precision')
tr_mio = tf.keras.metrics.MeanIoU(name='tr_mio', num_classes=3)

def training(neuralnet, dataset, epochs, batch_size):
    print("\nTraining to %d epochs (%d of minibatch size)" % (epochs, batch_size))

    iteration = 0
    prev_loss = 0
    # tf.keras.utils.plot_model(neuralnet.resModel, to_file='ResNeSt.png', show_shapes=True)

    for epoch in range(epochs):
        # only useful if using a diminishing learning rate
        if neuralnet.learning_rate < 1e-5:
            break
        while True:
            # Get the data for the next batch
            x_tr, y_tr, terminator = dataset.next_train(batch_size)  # y_tr does not used in this prj.
            # Take a step from that batch
            loss, class_score = neuralnet.step(x=x_tr, y=y_tr, train=True)
            # Loss is 256x80 so reduce to 1 number
            loss = tf.reduce_sum(loss)
            iteration += 1
            tr_recall.update_state(y_tr, class_score)
            tr_precision.update_state(y_tr, class_score)
            tr_mio.update_state(y_tr, class_score)
            print('.', end='')
            if (iteration + 1) % 100 == 0:
                print()
            if terminator: break

            # neuralnet.save_params()
        print()
        print("Epoch [%d / %d] (%d iteration)  Loss:%.5f, Recall:%.5f, Precision:%.5f, IoU:%.5f"
              % (epoch, epochs, iteration, loss, tr_recall.result(), tr_precision.result(), tr_mio.result()))
        tr_recall.reset_states()
        tr_precision.reset_states()
        tr_mio.reset_states()
        if prev_loss == loss:
            print("Model is throwing a fit")
            print(class_score)
        prev_loss = loss
        if epoch % 5 == 0:
            test(neuralnet, dataset, epoch)
        # test(neuralnet, dataset, epoch)
        if iteration < 5000:
            neuralnet.learning_rate = 1e-3
        elif iteration < 10000:
            if neuralnet.learning_rate == 4e-3:
                print("learning rate --> 1e-3")
            neuralnet.learning_rate = 1e-3
        elif iteration < 20000:
            if neuralnet.learning_rate == 1e-3:
                print("learning rate --> 3e-4")
            neuralnet.learning_rate = 3e-4
        elif iteration < 30000:
            if neuralnet.learning_rate == 3e-4:
                print("learning rate --> 1e-4")
            neuralnet.learning_rate = 1e-4
        else:
            if neuralnet.learning_rate == 1e-4:
                print("learning rate --> 1e-5")
            neuralnet.learning_rate = 1e-5


def test(neuralnet, dataset, epoch):
    print("\nTest...")

    # Much of this code is copied from the ResNest source I found and then translated to Keras
    # This simply calculates the metrics listed at the top using the logits and true
    # I do not know if this is 100% bug free since switching to probability labels
    total_loss = 0
    while True:
        x_te, y_te, terminator = dataset.next_test(1)  # y_te does not used in this prj.
        loss, class_score = neuralnet.step(x=x_te, y=y_te, train=False)
        loss = tf.reduce_sum(loss)
        precision.update_state(y_te, class_score)
        recall.update_state(y_te, class_score)
        pre_c2.update_state(y_te[:, :, -1], class_score[:, :, -1])
        re_c2.update_state(y_te[:, :, -1], class_score[:, :, -1])
        mio.update_state(y_te, class_score)
        total_loss += loss
        if terminator: break

    # This half of the code prints the metrics to the screen and saves them to a log file.
    # Later, you can open them up on tensorboard to see the progress.
    total_loss /= dataset.num_te
    with summary_writer.as_default():
        tf.summary.scalar("loss", total_loss, step=epoch)
        print("loss = {}".format(total_loss))
        f1 = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
        tf.summary.scalar("mean_IoU", mio.result(), step=epoch)
        print("IoU = {}".format(mio.result()))
        mio.reset_states()
        tf.summary.scalar("val_f1", f1, step=epoch)
        print("f1 = {}".format(f1))
        tf.summary.scalar("val_precision", precision.result(), step=epoch)
        print("precision = {}".format(precision.result()))
        precision.reset_states()
        tf.summary.scalar("recall_recall", recall.result(), step=epoch)
        print("recall = {}".format(recall.result()))
        precision.reset_states()


        # This is likely the same thing as above but I try to calculate the numbers for just class 2.
        # Comment out if you are that concerned about efficiency of the model's training.
        f1_2 = 2 * (pre_c2.result() * re_c2.result()) / (pre_c2.result() + re_c2.result())
        tf.summary.scalar("c2_f1", f1_2, step=epoch)
        tf.summary.scalar("c2_precision", pre_c2.result(), step=epoch)
        precision.reset_states()
        tf.summary.scalar("c2_recall", re_c2.result(), step=epoch)
        precision.reset_states()

        tf.summary.scalar("loss", total_loss, step=epoch)
        return f1


class ResNest(tf.Module):
    # I'm still having issues figuring out what I can put in the init and what I can't.
    # Most of this is self-explanatory. The class_factor and alpha are optional parameters in the loss function
    def __init__(self, height, width, channel, ksize, radix=4, kpaths=4):
        # print("\nInitializing Short-ResNeSt...")
        super(ResNest, self).__init__()
        self.height, self.width, self.channel = height, width, channel
        self.ksize = ksize
        self.radix, self.kpaths = radix, kpaths
        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=1, padding='SAME', kernel_regularizer=wDecay,
                                            kernel_initializer=tf.keras.initializers.HeNormal(),)
        self.conv1_act = tf.keras.layers.LeakyReLU()
        self.convtmp_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', kernel_regularizer=wDecay,
                                                kernel_initializer=tf.keras.initializers.HeNormal(),)
        self.convtmp_1bn = tf.keras.layers.BatchNormalization()
        self.convtmp_1act = tf.keras.layers.LeakyReLU()
        self.convtmp_2 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='SAME', kernel_regularizer=wDecay,
                                                kernel_initializer=tf.keras.initializers.HeNormal(),)
        self.convtmp_2bn = tf.keras.layers.BatchNormalization()
        self.convtmp_2act = tf.keras.layers.LeakyReLU()
        self.conv1_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv2_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv3_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv4_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv_1 = residual_S(ksize=self.ksize, outchannel=64, radix=self.radix, kpaths=self.kpaths)
        self.conv_2 = residual_S(ksize=self.ksize, outchannel=128, radix=self.radix, kpaths=self.kpaths)
        self.conv_3 = residual_S(ksize=self.ksize, outchannel=128, radix=self.radix, kpaths=self.kpaths)
        self.conv_4 = residual_S(ksize=self.ksize, outchannel=128, radix=self.radix, kpaths=self.kpaths)
        # block_channels = [64, 128, 256]
        # self.concats_1 = None
        # self.initialize = self.forward(np.zeros([1, 256, 80, 10]))

        # self.resModel = self.model()

    # This is the much larger model that is using full blocks on the decoder side.
    # @tf.function
    def forward(self, x):

        # img_input = tf.keras.layers.Input(shape=[self.height, self.width, self.channel])
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
        # resNest = tf.keras.Model(inputs=img_input, outputs=[result, [result, conv2_2, conv2_1]])
        return x_4, [x_3, x_2, x_1]

    def __call__(self, x, *args, **kwargs):
        return self.forward(x)

class residual_S(tf.Module):
    def __init__(self, ksize, outchannel, radix, kpaths, atrous=1):
        super(residual_S, self).__init__()
        self.kpaths = kpaths
        self.atrous = atrous
        self.ksize = ksize
        self.outchannel = outchannel
        self.radix = radix

        self.cardinal_blocks = []
        for _ in range(self.kpaths):
            cardinal_block = cardinal(self.ksize, self.outchannel // 2, self.radix, self.kpaths, atrous=self.atrous)
            self.cardinal_blocks.append(cardinal_block)

        self.concats_2 = tf.keras.layers.Conv2D(self.outchannel, self.ksize, strides=1, padding='SAME',
                                                dilation_rate=(self.atrous, self.atrous),
                                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                                kernel_regularizer=wDecay)

        self.convtmp_sc = tf.keras.layers.Conv2D(self.outchannel, 1, strides=1, padding='SAME',
                                                 dilation_rate=(self.atrous, self.atrous),
                                                 kernel_initializer=tf.keras.initializers.HeNormal(),
                                                 kernel_regularizer=wDecay)
        self.convtmp_scbn = tf.keras.layers.BatchNormalization()
        self.convtmp_scact = tf.keras.layers.LeakyReLU()

    def forward(self, x):
        concats_1 = None
        for layer_block in self.cardinal_blocks:
            cardinal = layer_block(x)
            if concats_1 is None:
                concats_1 = cardinal
            else:
                concats_1 = tf.concat([concats_1, cardinal], axis=3)

        concats_2 = self.concats_2(concats_1)
        x = self.convtmp_sc(x)
        x = self.convtmp_scbn(x)
        x = self.convtmp_scact(x)
        output = x + concats_2

        # if verbose: print(name, output.shape)
        return output

    def __call__(self, x, *args, **kwargs):
        return self.forward(x)

class cardinal(tf.Module):
    def __init__(self, ksize, outchannel, radix, kpaths, atrous=1):
        super(cardinal, self).__init__()
        self.outchannel = outchannel
        self.ksize = ksize
        self.radix = radix
        self.kpaths = kpaths
        self.atrous = atrous

        outchannel_cv11 = int(self.outchannel / self.radix / self.kpaths)
        outchannel_cvkk = int(self.outchannel / self.kpaths)
        self.conv1 = tf.keras.layers.Conv2D(outchannel_cv11, 1, strides=1, padding='SAME', dilation_rate=(self.atrous, self.atrous),
                                           kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=wDecay)
        self.conv1_bn = tf.keras.layers.BatchNormalization()
        self.conv1_act = tf.keras.layers.LeakyReLU()

        self.conv2 = tf.keras.layers.Conv2D(outchannel_cvkk, self.ksize,
                                           strides=1, padding='SAME', dilation_rate=(self.atrous, self.atrous),
                                           kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=wDecay)
        self.conv2_bn = tf.keras.layers.BatchNormalization()
        self.conv2_act = tf.keras.layers.LeakyReLU()
        self.split = split_attention(outchannel_cvkk, self.radix, self.atrous)

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
    def __init__(self, inchannel, radix, atrous=1):
        super(split_attention, self).__init__()
        self.inchannel = inchannel
        self.atrous = atrous
        self.radix = radix
        self.dense1 = tf.keras.layers.Conv2D(self.inchannel // 2, 1, strides=1, padding='SAME',
                                             dilation_rate=(self.atrous, self.atrous),
                                             kernel_initializer=tf.keras.initializers.HeNormal(),
                                             kernel_regularizer=wDecay)
        self.dense1_bn = tf.keras.layers.BatchNormalization()
        self.dense1_act = tf.keras.layers.LeakyReLU()
        self.dense2 = tf.keras.layers.Conv2D(self.inchannel, 1, strides=1, padding='SAME',
                                             dilation_rate=(self.atrous, self.atrous),
                                             kernel_initializer=tf.keras.initializers.HeNormal(),
                                             kernel_regularizer=wDecay)

    def forward(self, inputs):
        input_holder = None
        for idx_i, input in enumerate(inputs):
            if idx_i == 0:
                input_holder = input
            else:
                input_holder += input

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

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_data = '/DATA/TBI/Datasets/NPFiles/DispBal/TrainingData.npy'
    val_data = '/DATA/TBI/Datasets/NPFiles/DispBal/TestingData.npy'
    dataset = Dataset(train_data, val_data)
    # config = tf.estimator.RunConfig(train_distribute=mirrored_strategy)
    neuralnet = VisionTransformer()
    tf.keras.utils.model_to_dot(neuralnet.visionModel, to_file='TransUNet_dot.png', show_shapes=True)

    print(neuralnet.visionModel.summary())
    print(len(neuralnet.visionModel.layers))
    training(neuralnet=neuralnet, dataset=dataset, epochs=51, batch_size=16)
    neuralnet.visionModel.save('/DATA/TBI/Datasets/Models/ResNeSt_T1')


if __name__ == '__main__':

    main()
