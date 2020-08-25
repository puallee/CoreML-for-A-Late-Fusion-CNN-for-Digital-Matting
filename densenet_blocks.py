# -*- coding: utf-8 -*-
"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import Add
from keras.layers import UpSampling2D
from keras.layers import Lambda

#from keras.layers import BatchNormalization
#from util import BatchNorm, BilinearUpsampling
from keras.layers import BatchNormalization  as BatchNorm
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D

def dense_block(x, blocks, name, train_bn):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1), train_bn=train_bn)
    return x


def transition_block(x, reduction, name, train_bn):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNorm(axis=bn_axis, epsilon=1.001e-5,
                    name=name + '_bn')(x, training=train_bn)
    x = Activation('relu', name=name + '_relu')(x)
    skip = x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return skip, x

def conv_block(x, growth_rate, name, train_bn):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.

    import tensorflow as tf
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNorm(axis=bn_axis, epsilon=1.001e-5,
                    name=name + '_0_bn')(x, training=train_bn)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNorm(axis=bn_axis, epsilon=1.001e-5,
                    name=name + '_1_bn')(x1, training=train_bn)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x1)
   # x1 = Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC'))(x1)
    x1 = Conv2D(growth_rate, 3, padding='valid', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet_encoder(blocks,
             input_tensor,
             pooling=None,
             train_bn=False):
    """Instantiates the DenseNet architecture."""

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(input_tensor)
   # x = Lambda(lambda x: tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], mode='SYMMETRIC'))(input_tensor)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNorm(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn', )(x, training=train_bn)
    R1 = x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    R2 = x = dense_block(x, blocks[0], name='conv2', train_bn=train_bn)
    _, x = transition_block(x, 0.5, name='pool2', train_bn=train_bn)
    R3 = x = dense_block(x, blocks[1], name='conv3', train_bn=train_bn)
    _, x = transition_block(x, 0.5, name='pool3', train_bn=train_bn)
    R4 = x = dense_block(x, blocks[2], name='conv4', train_bn=train_bn)
    _, x = transition_block(x, 0.5, name='pool4', train_bn=train_bn)
    x = dense_block(x, blocks[3], name='conv5', train_bn=train_bn)

    x = BatchNorm(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x, training=train_bn)

    if pooling == 'avg':
        x = AveragePooling2D(7, name='avg_pool')(x)
    elif pooling == 'max':
        x = MaxPooling2D(7, name='max_pool')(x)

    return [R1, R2, R3, R4], x

def fpn_side_output_block(deconv_input, upsample_input, block_name,
                          network_name, train_bn, use_bias, output_shape,
                          up_output=True, out_output=True, d=128):
    prefix = network_name + '_side_' + block_name
    if deconv_input.shape[3] == d:
        x = deconv_input
    else:
        x = Conv2D(d, (1, 1), strides=(1, 1), padding='same',
                      name=prefix + '_conv1', use_bias=use_bias)(deconv_input)

    if upsample_input != None:
        x = Add(name=prefix+'_add')([x, upsample_input])
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
       # x = Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC'))(x)
        add = x = Conv2D(d, (3, 3), padding='valid',
                            name=prefix+'_conv2', use_bias=use_bias)(x)
    else:
        add = None
    
    if out_output:
        out = Conv2D(1, (1, 1), strides=(1, 1), padding='same',
                      name=prefix + '_conv3', use_bias=use_bias)(x)
        out = Activation('sigmoid', name=prefix+'_sigmoid')(out)
        # out = BilinearUpsampling(output_size=(output_shape[0], output_shape[1]),
        #                              name=prefix+'_out')(out)
        # out = UpSampling2D(data_format=K.image_data_format(),size=(1, 1),
        #                       name=prefix+'_up')(out)                             
    else:
        out = None

    if up_output:
        up = UpSampling2D(data_format=K.image_data_format(),
                             name=prefix+'_up')(x)
    else:
        up = None

    return add, up, out

def deconv_block(x, skip, network_name, fpn_d, train_bn):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = UpSampling2D(2, data_format=K.image_data_format())(x)
    if not skip is None:
        channel = K.int_shape(skip)[bn_axis]
        channel = fpn_d if channel < fpn_d else channel
        #x = Conv2D(channel, (1, 1), name=network_name+'_conv',
                   #padding='same', use_bias=False)(x)
        #x = Add(name=network_name+'_add')([x, skip])
        x = Concatenate(axis=bn_axis)([x, skip])
        x = Conv2D(channel, (1, 1), name=network_name+'_conv',
                   padding='same', use_bias=False)(x)
    else:
        channel = K.int_shape(x)[bn_axis]
        channel = fpn_d if channel < fpn_d else channel
        x = Conv2D(channel, (1, 1), name=network_name+'_conv', use_bias=False)(x)
    
    x = BatchNorm(axis=bn_axis, epsilon=1.001e-5,
                    name=network_name+'_bn')(x, training=train_bn)
    x = Activation('relu', name=network_name+'_relu')(x)
    return x

def DenseNet_decoder(input_tensor,
             skip_connection,
             network_name,
             output_dim,
             fpn_d,
             train_bn=False):
    """Instantiates the DenseNet architecture. (decoder part) """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    R1, R2, R3, R4 = skip_connection
    DC4 = x = deconv_block(input_tensor, R4, fpn_d=fpn_d,
                network_name=network_name+'_deconv5', train_bn=train_bn)
    DC3 = x = deconv_block(x, R3, fpn_d=fpn_d,
                network_name=network_name+'_deconv4', train_bn=train_bn)
    DC2 = x = deconv_block(x, R2, fpn_d=fpn_d,
                network_name=network_name+'_deconv3', train_bn=train_bn)
    DC1 = x = deconv_block(x, R1, fpn_d=fpn_d,
                network_name=network_name+'_deconv2', train_bn=train_bn)
    # We should get 256*256*64 at DC1
    x = deconv_block(x, None, network_name=network_name+'_deconv1', fpn_d=fpn_d, train_bn=train_bn)
    
    # FPN
    up4 = Conv2D(fpn_d, (1, 1), padding='same', name=network_name+'_up4_conv', use_bias=True)(DC4)
    up4 = Activation('relu', name=network_name+'_up4_relu')(up4)
    up4 = UpSampling2D(data_format=K.image_data_format(), name=network_name+'_up4_up')(up4)
    [_, up3, out_8] = fpn_side_output_block(DC3, up4, block_name='3', output_shape=output_dim,
                                                 network_name=network_name, d=fpn_d,
                                                 train_bn=train_bn, use_bias=True)
    [_, up2, out_4] = fpn_side_output_block(DC2, up3, block_name='2', output_shape=output_dim,
                                            network_name=network_name, d=fpn_d,
                                            train_bn=train_bn, use_bias=True)
    [_, up1, out_2] = fpn_side_output_block(DC1, up2, block_name='1', output_shape=output_dim,
                                            network_name=network_name, d=fpn_d,
                                            train_bn=train_bn, use_bias=True)
    [add0, _, out] = fpn_side_output_block(x, up1, block_name='0', up_output=False, d=fpn_d,
                                        network_name=network_name, output_shape=output_dim,
                                        train_bn=train_bn, use_bias=True)

    return [out, out_2, out_4, out_8, add0]
