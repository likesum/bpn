import functools

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

Conv2D = functools.partial(layers.Conv2D, activation='relu', padding='same')

class BPN:
    def __init__(self, num_basis=90, ksz=15, burst_length=8, color=False):
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length
        self.color = color

        nchannels = burst_length * 6 if color else burst_length * 2
        inputs = keras.Input(shape=(None, None, nchannels))
        outputs = self.forward(inputs)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def down_block(self, out, nchannel, pfx=''):
        out = Conv2D(nchannel, 3, name=pfx+'_1')(out)
        skip = Conv2D(nchannel, 3, name=pfx+'_2')(out)
        out = layers.MaxPool2D((2, 2), strides=(2, 2))(skip)
        return out, skip

    def coeff_up_block(self, out, nchannel, skip, pfx=''):
        # out = tf.image.resize(out, 2*tf.shape(out)[1:3])
        out = tf.compat.v1.image.resize_bilinear(out, 2*tf.shape(out)[1:3])
        out = Conv2D(nchannel, 3, name=pfx+'_1')(out)
        out = layers.concatenate([out, skip], axis=-1)
        out = Conv2D(nchannel, 3, name=pfx+'_2')(out)
        out = Conv2D(nchannel, 3, name=pfx+'_3')(out)
        return out

    def basis_up_block(self, out, nchannel, skip, pfx=''):
        shape = tf.shape(out)
        # out = tf.image.resize(out, 2*shape[1:3])
        out = tf.compat.v1.image.resize_bilinear(out, 2*tf.shape(out)[1:3])
        out = Conv2D(nchannel, 3, name=pfx+'_1')(out)
        
        # pooled skip connections
        skip = tf.reduce_mean(skip, axis=[1,2], keepdims=True)
        skip = tf.tile(skip, [1,2*shape[1],2*shape[2],1])
        out = layers.concatenate([out, skip], axis=-1)

        out = Conv2D(nchannel, 3, name=pfx+'_2')(out)
        out = Conv2D(nchannel, 3, name=pfx+'_3')(out)
        return out

    def encoder(self, out):
        out = Conv2D(64, 3, name='inp')(out)

        out, d1 = self.down_block(out, 64, 'down1')
        out, d2 = self.down_block(out, 128, 'down2')
        out, d3 = self.down_block(out, 256, 'down3')
        out, d4 = self.down_block(out, 512, 'down4')
        out, d5 = self.down_block(out, 1024, 'down5')

        out = Conv2D(
            1024, 3, name='bottleneck1')(out)
        out = Conv2D(
            1024, 3, name='bottleneck2')(out)
        return out, [d5, d4, d3, d2, d1]

    def coeff_decoder(self, out, skips):
        out = self.coeff_up_block(out, 512, skips[0], 'coeff_up1')
        out = self.coeff_up_block(out, 256, skips[1], 'coeff_up2')
        out = self.coeff_up_block(out, 128, skips[2], 'coeff_up3')
        out = self.coeff_up_block(out, 64, skips[3], 'coeff_up4')
        out = self.coeff_up_block(out, 64, skips[4], 'coeff_up5')
        out = Conv2D(64, 3, name='coeff_end1')(out)
        out = Conv2D(64, 3, name='coeff_end2')(out)
        out = Conv2D(self.num_basis, 3, activation=None, name='coeff_end3')(out)
        out = layers.Softmax()(out)
        return out

    def basis_decoder(self, out, skips):
        assert self.ksz == 15
        out = tf.reduce_mean(out, axis=[1,2], keepdims=True) # 1x1
        out = self.basis_up_block(out, 512, skips[0], 'basis_up1') # 2x2
        out = self.basis_up_block(out, 256, skips[1], 'basis_up2') # 4x4
        out = self.basis_up_block(out, 256, skips[2], 'basis_up3') # 8x8
        out = self.basis_up_block(out, 128, skips[3], 'basis_up4') # 16x16
        out = Conv2D(128, 2, padding='valid', name='basis_end1')(out)
        out = Conv2D(128, 3, name='basis_end2')(out)

        if self.color:
            out = Conv2D(
                self.burst_length * 3 * self.num_basis, 3,
                activation=None, name='basis_end3')(out)
            out = layers.Reshape(
                (self.ksz**2 * self.burst_length, 3, self.num_basis))(out)
            out = layers.Softmax(axis=-3)(out)
            out = layers.Reshape(
                (self.ksz**2 * self.burst_length * 3, self.num_basis))(out)
            out = tf.transpose(out, [0, 2, 1])
        else:
            out = Conv2D(
                self.burst_length * self.num_basis, 3,
                activation=None, name='basis_end3')(out)
            out = layers.Reshape(
                (self.ksz**2 * self.burst_length, self.num_basis))(out)
            out = layers.Softmax(axis=-2)(out)
            out = tf.transpose(out, [0, 2, 1])

        return out

    def forward(self, inputs):
        features, skips = self.encoder(inputs)

        basis = self.basis_decoder(features, skips)
        coeffs = self.coeff_decoder(features, skips)
        return basis, coeffs

