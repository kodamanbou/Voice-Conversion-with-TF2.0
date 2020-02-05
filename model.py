import tensorflow as tf
import tensorflow_addons as tfa
import hyperparameter as hp


class CycleGAN2(tf.keras.Model):
    def __init__(self):
        super(CycleGAN2, self).__init__()
        self.generatorA2B = Generator()
        self.generatorB2A = Generator()
        self.discriminator_A = PatchGanDiscriminator()
        self.discriminator_B = PatchGanDiscriminator()
        self.discriminator_A_dot = PatchGanDiscriminator()  # Cycle A
        self.discriminator_B_dot = PatchGanDiscriminator()  # Cycle B

    def __call__(self, inputs, training=None, mask=None):
        real_A = tf.cast(inputs[0], tf.float32)
        real_B = tf.cast(inputs[1], tf.float32)
        outputs = []

        generation_B = self.generatorA2B(real_A)
        generation_A = self.generatorB2A(real_B)
        cycle_A = self.generatorB2A(generation_B)
        cycle_B = self.generatorA2B(generation_A)
        identity_A = self.generatorB2A(real_A)
        identity_B = self.generatorA2B(real_B)
        outputs += [generation_A, generation_B, cycle_A, cycle_B, identity_A, identity_B]

        discrimination_A_real = self.discriminator_A(real_A)
        discrimination_A_fake = self.discriminator_A(generation_A)
        discrimination_B_real = self.discriminator_B(real_B)
        discrimination_B_fake = self.discriminator_B(generation_B)
        outputs += [discrimination_A_real, discrimination_A_fake, discrimination_B_real, discrimination_B_fake]

        discrimination_A_dot_real = self.discriminator_A_dot(real_A)
        discrimination_A_dot_fake = self.discriminator_A_dot(cycle_A)
        discrimination_B_dot_real = self.discriminator_B_dot(real_B)
        discrimination_B_dot_fake = self.discriminator_B_dot(cycle_B)
        outputs += [discrimination_A_dot_real, discrimination_A_dot_fake, discrimination_B_dot_real,
                    discrimination_B_dot_fake]

        return outputs


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.h1 = tf.keras.layers.Conv2D(128, kernel_size=(5, 15), padding='same', name='h1_conv')
        self.h1_gates = tf.keras.layers.Conv2D(128, kernel_size=(5, 15), padding='same', name='h1_conv_gates')
        self.h1_glu = tf.keras.layers.Multiply(name='h1_glu')

        self.d1 = Downsample2DBlock(256, kernel_size=(5, 5), strides=2, name_prefix='downsample2d_block1_')
        self.d2 = Downsample2DBlock(256, kernel_size=(5, 5), strides=2, name_prefix='downsample2d_block2_')

        self.resh1 = tf.keras.layers.Conv1D(256, kernel_size=1, strides=1, padding='same', name='resh1_conv')
        self.resh1_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name='resh1_norm')

        self.res1 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block1_')
        self.res2 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block2_')
        self.res3 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block3_')
        self.res4 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block4_')
        self.res5 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block5_')
        self.res6 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block6_')

        self.resh2 = tf.keras.layers.Conv1D(2304, kernel_size=1, padding='same', name='resh2_conv')
        self.resh2_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name='resh2_norm')

        # upsampling
        self.u1 = Upsample2DBlock(filters=1024, kernel_size=5, name_prefix='upsampling2d_block1_')
        self.u2 = Upsample2DBlock(filters=512, kernel_size=5, name_prefix='upsampling2d_block2_')

        self.conv_out = tf.keras.layers.Conv2D(1, kernel_size=(5, 15), padding='same', name='conv_out')

    def __call__(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        h1 = self.h1(inputs)
        h1_gates = self.h1_gates(inputs)
        h1_glu = self.h1_glu([h1, tf.sigmoid(h1_gates)])

        d1 = self.d1(h1_glu)
        d2 = self.d2(d1)
        d3 = tf.squeeze(tf.reshape(d2, shape=(hp.batch_size, 1, -1, 2304)), axis=1)
        resh1 = self.resh1(d3)
        resh1_norm = self.resh1_norm(resh1)

        res1 = self.res1(resh1_norm)
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        res6 = self.res6(res5)

        resh2 = self.resh2(res6)
        resh2_norm = self.resh2_norm(resh2)
        resh3 = tf.reshape(tf.expand_dims(resh2_norm, axis=1), shape=(hp.batch_size, 9, -1, 256))

        u1 = self.u1(resh3)
        u2 = self.u2(u1)
        conv_out = self.conv_out(u2)
        out = tf.squeeze(conv_out, axis=-1)

        return out


class PatchGanDiscriminator(tf.keras.Model):
    def __init__(self):
        super(PatchGanDiscriminator, self).__init__()
        self.h1 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation=tf.nn.leaky_relu,
                                         name='h1_conv')
        self.h1_gates = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation=tf.nn.leaky_relu,
                                               name='h1_conv_gates')
        self.h1_glu = tf.keras.layers.Multiply(name='h1_glu')

        self.d1 = Downsample2DBlock(256, kernel_size=(3, 3), strides=2, activation=tf.nn.leaky_relu,
                                    name_prefix='downsample2d_block1_')
        self.d2 = Downsample2DBlock(512, kernel_size=(3, 3), strides=2, activation=tf.nn.leaky_relu,
                                    name_prefix='downsample2d_block2_')
        self.d3 = Downsample2DBlock(1024, kernel_size=(3, 3), strides=2, activation=tf.nn.leaky_relu,
                                    name_prefix='downsample2d_block3_')
        self.d4 = Downsample2DBlock(1024, kernel_size=(1, 5), strides=1, activation=tf.nn.leaky_relu,
                                    name_prefix='downsample2d_block4_')

        self.out = tf.keras.layers.Conv2D(1, kernel_size=(1, 3), strides=1, padding='same', activation=tf.nn.leaky_relu,
                                          name='out_conv')

    def __call__(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)  # [N, M, T, 1]
        h1 = self.h1(inputs)
        h1_gates = self.h1_gates(inputs)
        h1_glu = self.h1_glu([h1, tf.sigmoid(h1_gates)])

        d1 = self.d1(h1_glu)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        out = self.out(d4)

        return out


class Downsample2DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, activation=None, name_prefix=None):
        super(Downsample2DBlock, self).__init__()
        self.h1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=activation,
                                         padding='same', name=name_prefix + 'h1_conv')
        self.h1_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')
        self.h1_gates = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=activation,
                                               padding='same', name=name_prefix + 'h1_gates')
        self.h1_norm_gates = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')
        self.h1_glu = tf.keras.layers.Multiply(name=name_prefix + 'h1_glu')

    def __call__(self, inputs, training=None, mask=None):
        h1 = self.h1(inputs)
        h1 = self.h1_norm(h1)
        gates = self.h1_gates(inputs)
        gates = self.h1_norm_gates(gates)
        h1_glu = self.h1_glu([h1, tf.sigmoid(gates)])
        return h1_glu


class Residual1DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, name_prefix=None):
        super(Residual1DBlock, self).__init__()
        self.h1 = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, strides=strides,
                                         padding='same', name=name_prefix + 'h1_conv')
        self.h1_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')
        self.h1_gates = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, strides=strides,
                                               padding='same', name=name_prefix + 'h1_gates')
        self.h1_norm_gates = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')
        self.h1_glu = tf.keras.layers.Multiply(name=name_prefix + 'h1_glu')

        self.h2 = tf.keras.layers.Conv1D(filters // 2, kernel_size=kernel_size, strides=strides,
                                         padding='same', name=name_prefix + 'h2_conv')
        self.h2_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')

    def __call__(self, inputs, training=None, mask=None):
        h1 = self.h1(inputs)
        h1 = self.h1_norm(h1)
        h1_gates = self.h1_gates(inputs)
        h1_gates = self.h1_norm_gates(h1_gates)
        h1_glu = self.h1_glu([h1, tf.sigmoid(h1_gates)])

        h2 = self.h2(h1_glu)
        h2_norm = self.h2_norm(h2)

        h3 = inputs + h2_norm
        return h3


class Upsample2DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, name_prefix=None):
        super(Upsample2DBlock, self).__init__()
        self.h1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                         padding='same', name=name_prefix + 'h1_conv')
        self.h1_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')
        self.h1_gates = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                               padding='same', name=name_prefix + 'h1_gates')
        self.h1_norm_gates = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')
        self.h1_glu = tf.keras.layers.Multiply(name=name_prefix + 'h1_glu')

    def __call__(self, inputs, training=None, mask=None):
        h1 = self.h1(inputs)
        h1_shuffle = tf.nn.depth_to_space(h1, block_size=2, name='h1_shuffle')
        h1_norm = self.h1_norm(h1_shuffle)
        h1_gates = self.h1_gates(inputs)
        h1_shuffle_gates = tf.nn.depth_to_space(h1_gates, block_size=2, name='h1_shuffle_gates')
        h1_norm_gates = self.h1_norm_gates(h1_shuffle_gates)
        h1_glu = self.h1_glu([h1_norm, tf.sigmoid(h1_norm_gates)])
        return h1_glu
