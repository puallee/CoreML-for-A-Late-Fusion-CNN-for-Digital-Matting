import os
import random
import math
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow.compat.v1 as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
#from util import Subone
from keras.layers import BatchNormalization  as BatchNorm
from keras.layers.core import Layer
from keras.layers.core import Layer

############################################################
#  Fusion Net
############################################################
class Subone(Layer):
     def __init__(self,**kwargs):
        super().__init__(**kwargs)
     def call(self, inputs):
        var=K.ones_like(inputs)
        return var-inputs

def fusion_graph_with_rgb(fg_features, bg_features, input_rgb, output_shape=None, filters=[256, 128, 128, 64, 256],
                          train_bn=True, network_name='fusion_', d=256):
    '''

    :param fg_features: [b, 512, 512, d]
    :param bg_features: [b, 512, 512, d]
    :param trimap: [b, 512, 512, 1]
    :param backbone_r1: [b, 256, 256, 64]
    :param filters:
    :param train_bn:
    :param network_name:
    :param broadcast_trimap:
    :return:
    '''

    if len(filters) == 5:
        nb_filter0, nb_filter1, nb_filter2, nb_filter3, nb_filter4 = filters
    else:
        nb_filter0, nb_filter1, nb_filter2, nb_filter3 = filters

    # TODO: input_rgb = raw_image - mean_pixel.
    # TODO: BatchNorm
    conv_rgb = KL.Conv2D(d, (3, 3), strides=1, name=network_name+'convrgb', padding='same')(input_rgb)
    conv_rgb = BatchNorm(name=network_name + "bnrgb")(conv_rgb, training=train_bn)
    fusion_input = KL.Concatenate(axis=3, name=network_name+'input_concate')([fg_features, bg_features, conv_rgb])

    x = KL.Conv2D(nb_filter0, (3, 3), strides=1, name=network_name + "conv0", padding='same')(fusion_input)
    x = BatchNorm(name=network_name + "bn0")(x, training=train_bn)
    conv0 = KL.Activation('relu', name=network_name + "relu0")(x)

    # fusion_conv1
    x = KL.Conv2D(nb_filter1, (3, 3), strides=1, name=network_name + "conv1", padding='same')(conv0)
    x = BatchNorm(name=network_name + "bn1")(x, training=train_bn)
    x = KL.Activation('relu', name=network_name + "relu1")(x)
    conv1 = x

    # fusion_conv2
    x = KL.Conv2D(nb_filter2, (3, 3), strides=1, name=network_name + "conv2", padding='same')(x)
    x = BatchNorm(name=network_name + "bn2")(x, training=train_bn)
    x = KL.Activation('relu', name=network_name + 'relu2')(x)
    conv2 = x

    # fusion_conv3
    x = KL.Conv2D(nb_filter3, (3, 3), strides=1, name=network_name + "conv3", padding='same')(x)
    x = BatchNorm(name=network_name + "bn3")(x, training=train_bn)
    x = KL.Activation('relu', name=network_name + 'relu3')(x)
    conv3 = x

    # fusion_conv4
    if len(filters) == 5:
        x = KL.Conv2D(nb_filter4, (3, 3), strides=1, name=network_name + "conv4", padding='same')(x)
        x = BatchNorm(name=network_name + "bn4")(x, training=train_bn)
        x = KL.Activation('relu', name=network_name + 'relu4')(x)
        conv4 = x

    # fusion_output
    x = KL.Conv2D(1, (1, 1), strides=1, name=network_name + "conv_output", padding='same')(x)
    # x = BatchNorm(name=network_name+"bn_output")(x, training=train_bn)
    output = KL.Activation('sigmoid', name=network_name + "sigmoid_output")(x)

    # output = BilinearUpsampling(output_size=(output_shape[0], output_shape[1]), name=network_name + '_upsampling')(
    #     output)
    return output, [conv0, conv1, conv2, conv3]

def blending_graph(fg_out, bg_out, fg_weights, network_name='fusion_'):
    # bg_weights = KL.Subtract()([K.constant(K.ones_like(fg_weights)), fg_weights])

    weighted_fg = KL.Multiply(name=network_name + 'fg_mul')([fg_out, fg_weights])

   #temp_1 = KL.Lambda(lambda x: 1.0-x, name=network_name+'reverse_lambda_bg')(bg_out)
   #temp_1 = KL.Add(name=network_name + 'reverse_lambda_bg')([bg_out, -bg_out])
   #temp_2 = KL.Lambda(lambda x: 1.0-x, name=network_name+'reverse_lambda_blendingweight')(fg_weights)
   #temp_2 = KL.Add(name=network_name + 'reverse_lambda_blendingweigh')([var, -fg_weights])

    weighted_bg = KL.Multiply(name=network_name+'bg_mul')([bg_out, fg_weights])
    weighted_bg = KL.Subtract(name=network_name + 'addbg')([weighted_bg, bg_out])
    weighted_bg = KL.Subtract(name=network_name + 'addfg')([weighted_bg, fg_weights])

    #weighted_bg = KL.Multiply(name=network_name+'bg_mul')([temp_1, temp_2])

    final_result = KL.Add(name=network_name + 'blending_output')([weighted_fg, weighted_bg])
  #  final_result=KL.Reshape([ ,960,640])(final_result)
    return final_result
## python(keras+lambda:tensorflow) - coreml , lambda, comstant:

def generate_trimap(predict_tensors, edge_width=20, threshold=0.5):
    # binarize
    predict_fg_tensor = K.cast(predict_tensors[0] > threshold, dtype='float32')
    predict_bg_tensor = K.cast((1 - predict_tensors[1]) > threshold, dtype='float32')
    # heat = K.cast(tf.abs(fg-bg) > threshold, dtype='float32')

    # make trimap
    with tf.compat.v1.variable_scope('erosion_scope', reuse=tf.compat.v1.AUTO_REUSE):
        kernel = tf.compat.v1.get_variable('erosion_kernel', [edge_width, edge_width, 1],
                                 initializer=tf.compat.v1.zeros_initializer(), trainable=False)
        dilation_fg = tf.compat.v1.nn.dilation2d(predict_fg_tensor, filter=kernel,
                                       strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                       padding='SAME', name='fg_dilation')
        erosion_bg = tf.compat.v1.nn.erosion2d(predict_bg_tensor, kernel=kernel,
                                     strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                     padding='SAME', name='fg_erosion')
        edge_map = dilation_fg - erosion_bg
        edge_float = K.cast(edge_map > 0, dtype='float32')
        trimap = tf.where(edge_map > 0, edge_float * 0.5, predict_fg_tensor)

        trimap = tf.stop_gradient(trimap, name="stop_trimap_gradient")
        edge_float = tf.stop_gradient(edge_float, name="stop_trimask_gradient")

    return [trimap, edge_float]