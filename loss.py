import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
import keras.losses as losses
import keras.backend as K
import keras.layers as KL
    
def crossentropy_loss(output, seg_map, alpha_map, ce_weight):
    ce_weight = tf.squeeze(ce_weight, axis=-1)
    weight = tf.where(ce_weight > 0.5, tf.ones_like(ce_weight), 0.5 * tf.ones_like(ce_weight))
    crossentropy = weight * losses.binary_crossentropy(seg_map, output)
    return K.mean(crossentropy)
    
def l1_loss(output, seg_map, alpha_map, ce_weight):
    loss = losses.mean_absolute_error(alpha_map, output)
    return K.mean(loss)

def l2_loss(output, seg_map, alpha_map, ce_weight):
    loss = losses.mean_squared_error(alpha_map, output)
    return K.mean(loss)
    
def l1_l2_loss(output, seg_map, alpha_map, ce_weight):
    ce_weight = tf.squeeze(ce_weight, axis=-1)
    l1_loss = losses.mean_absolute_error(alpha_map, output)
    l2_loss = losses.mean_squared_error(alpha_map, output)
    loss = tf.where(ce_weight > 0.5, l2_loss, l1_loss)
    return K.mean(loss)
    
def segmentation_loss(output, seg_map, alpha_map, ce_weight):
    return tf.add(crossentropy_loss(output, seg_map, alpha_map, ce_weight),\
                  l1_l2_loss(output, seg_map, alpha_map, ce_weight))

def alpha_gradient_loss(matting_result, matting_gt):
    def image_gradients(image):
        if image.get_shape().ndims != 4:
            raise ValueError('image_gradents expects a 4D tensor [batch_size, h, w, d], not %s.', image.get_shape())
        image_shape = array_ops.shape(image)
        batch_size, height, width, depth = array_ops.unstack(image_shape)
        dy = image[:, 1:, :, :] - image[:, :-1, :, :]
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]

        # Return tensors with same size as original image by concatenating
        # zeros. Place the gradient [I(x+1, y)-I(x, y)] on the base pixel (x, y)
        shape = array_ops.stack([batch_size, 1, width, depth])
        dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
        dy = array_ops.reshape(dy, image_shape)

        shape = array_ops.stack([batch_size, height, 1, depth])
        dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
        dx = array_ops.reshape(dx, image_shape)

        return dy, dx

    gt_dy, gt_dx = image_gradients(matting_gt)
    predict_dy, predict_dx = image_gradients(matting_result)

    dy_loss_square = K.square(gt_dy - predict_dy)
    dx_loss_square = K.square(gt_dx - predict_dx)

    dy_loss = K.mean(K.sqrt(dy_loss_square + K.epsilon()))
    dx_loss = K.mean(K.sqrt(dx_loss_square + K.epsilon()))

    return tf.add(dx_loss, dy_loss)

def alpha_sparsity_regular_graph(matting_result):
    gamma = 0.9

    reg_1 = K.pow(matting_result+1e-5, gamma)
    reg_2 = K.pow(1-matting_result+1e-5, gamma)  # negative base will cause Nan
    reg_loss = KL.Add()([reg_1, reg_2])

    # _one = KL.Lambda(lambda x: tf.ones_like(x))(matting_result)
    #reg_loss = KL.Subtract()([reg_loss, _one])

    return K.mean(reg_loss)


def matting_weighted_l1_loss(matting_gt, blending_result, trimask):
    trimask = tf.where(trimask > 0.5, tf.ones_like(trimask), 0.1 * tf.ones_like(trimask))
    square = K.square(matting_gt - blending_result)
    l1 = K.sqrt(square + K.epsilon())
    weighted_l1 = KL.Multiply(name='fusion_trimask_l1_loss_mul')([trimask, l1])
    # return K.mean(weighted_l1)
    return K.sum(weighted_l1) / K.sum(trimask)
