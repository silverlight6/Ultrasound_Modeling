import tensorflow as tf
# import tensorflow_addons as tfa
from ResNest import ResNest
from Decoder import DecoderCup


class Attention(tf.Module):
    def __init__(self, num_heads=4, attention_head_size=512, attention_dropout_rate=0.0, wDecay=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = attention_head_size
        self.wDecay = wDecay
        self.qkv_size = attention_head_size // self.num_heads
        self.query = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=self.wDecay, dtype=tf.float32)
        self.key = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=self.wDecay, dtype=tf.float32)
        self.value = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=self.wDecay, dtype=tf.float32)
        self.out = tf.keras.layers.Dense(self.hidden_size, kernel_regularizer=self.wDecay, dtype=tf.float32)
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

    def __init__(self, img_size, hidden_size=512, dropout_rate=0.0, wDecay=None):
        super(Embeddings, self).__init__()
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.wDecay = wDecay
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
        # self.hybrid_model = SwinTransformer(model_name='swin_large_patch4_window7_384', img_size=self.img_size,
        #                                     in_chans=10)
        # Try using a reshape instead of a convolution later on.
        # self.patch_embeddings = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, padding='same',
        #                                                strides=patch_size, kernel_regularizer=wDecay)
        self.patch_embeddings = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=1, padding='valid',
                                                       strides=1, kernel_regularizer=self.wDecay)
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
    def __init__(self, hidden_size=512, wDecay=None):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        self.ffn = Mlp()
        self.attn = Attention(wDecay=wDecay)

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
    def __init__(self, xdim, ydim, num_layers=8, wDecay=None):
        super(Encoder, self).__init__()
        self.xDim = xdim
        self.yDim = ydim
        self.encoder_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        self.Transformer_layers = []
        for _ in range(num_layers):
            Transformer_layers = Block(wDecay=wDecay)
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
    def __init__(self, img_size, wDecay=None):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = Encoder(img_size[0], img_size[1], wDecay=wDecay)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features

    def __call__(self, input_ids, *args, **kwargs):
        return self.forward(input_ids)


class VisionTransformer(tf.Module):
    def __init__(self, batch_size, img_size=(256, 80), num_classes=3, learning_rate=1e-3, weight_decay=1e-4, ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.transformer = Transformer(img_size)
        self.decoder = DecoderCup(num_classes)
        self.input_shape = [img_size[0], img_size[1], 10]
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.learning_rate = learning_rate
        # self.optimizer = tfa.optimizers.AdamW(weight_decay=self.weight_decay, learning_rate=self.learning_rate)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1,
                                                            reduction=tf.keras.losses.Reduction.NONE)
        # self.loss = self.my_loss_cat
        self.alpha = 2
        self.class_factor = [0.06329, 0.027567, 0.90914]
        self.visionModel = self.model()
        # self.initialize = self.forward(np.zeros([4, 256, 80, 10]))

    def model(self):
        inputA = tf.keras.Input(shape=self.input_shape, batch_size=int(self.batch_size))
        output = self.forward(inputA)
        model = tf.keras.Model(inputs=inputA, outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        logits = self.decoder(x, features)
        return logits, attn_weights

    def compute_loss(self, y_true, y_pred):
        per_example_loss = self.loss(y_true, y_pred)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)

    '''
    jit_compile simply allows the model to use more of the GPU space available to it.
    This method is the training loop for the model.
    GradientTape records all of the partials for the model and backprops to update weights
    The bottom is a few calculations for metrics purposes.
    '''
    @tf.function(jit_compile=True)
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # # Really not sure if this line of code is helpful or just slows things down. Uncomment if you want.
            tape.watch(self.visionModel.trainable_variables)
            logits, _ = self.forward(x)
            smce = self.compute_loss(y_true=y, y_pred=logits)
            # smce += sum(self.visionModel.losses)
        gradients = tape.gradient(smce, self.visionModel.trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(clip_gradients, self.visionModel.trainable_variables))
        return smce, logits

    @tf.function(jit_compile=True)
    def step(self, x, y):
        # # Really not sure if this line of code is helpful or just slows things down. Uncomment if you want.
        logits, _ = self.forward(x)
        smce = self.compute_loss(y_true=y, y_pred=logits)
        # smce += sum(self.visionModel.losses)
        return smce, logits

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
        CE = -3 * tf.reduce_sum(y_true * tf.cast(tf.math.log(y_pred), tf.float32) * self.class_factor, axis=[0, 3])

        # CE = -tf.reduce_sum(y_true * tf.cast(tf.math.log(y_pred), tf.float32) *
        #       tf.pow(1.0 - y_pred, self.alpha), axis=[0, 3])
        # CE = -3 * tf.reduce_sum(y_true * tf.cast(tf.math.log(y_pred), tf.float32), axis=3)
        # CE += randOffSet * 0.1

        # -3 if using class factor, -1 otherwise. (class factor in my implementation is divides by number of classes)
        # tf.print(CE)
        return CE
