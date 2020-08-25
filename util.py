import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.callbacks as KC
#from keras.engine.topology import preprocess_weights_for_loading
from tensorflow.python.keras.saving.hdf5_format import preprocess_weights_for_loading
from keras.utils import plot_model, conv_utils
from keras.layers.core import Layer
#from keras.backend.common import normalize_data_format

############################################################
#  Bilinear Upsampling Layer
############################################################
class Subone(Layer):
     def __init__(self,**kwargs):
        super().__init__(**kwargs)
     def call(self, inputs):
        var=K.ones_like(inputs)
        return var-inputs

class BilinearUpsampling(KL.Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        #self.data_format = conv_utils.normalize_data_format(data_format)
        #self.data_format = K.backend.common.normalize_data_format(data_format)
        if keras.__version__ > "2.2.0":
            from keras.backend import normalize_data_format
            self.data_format = normalize_data_format(data_format)
        else:
            from keras.utils.conv_utils import normalize_data_format
            self.data_format = normalize_data_format(data_format)
        self.input_spec = KE.InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.compat.v1.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
           # return keras.layers.UpSampling2D(size=(self.upsampling[0], self.upsampling[1]))(inputs)
        else:
            return tf.compat.v1.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)
            #return keras.layers.UpSampling2D(size=(1, 1))(inputs)
    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

############################################################
#  Batch Normalization Wrapper
############################################################

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)
 

############################################################
#  Utility Functions
############################################################

def make_img_list(img_path):
    if img_path[-3:] == 'txt':
        img_list = []
        with open(img_path, 'r') as f:
            base = os.path.dirname(os.path.abspath(img_path))
            for line in f:
                item = []
                for path in line.split(' '):     
                    path = path.strip()
                    if not len(path):
                        continue
                    if not os.path.isabs(path):
                        path = os.path.join(base, path)
                    item.append(path)
                if len(item) != 0:
                    img_list.append(item)
    else:
        img_list = [[img_path]]
    
    return img_list

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    _, height, width, channel = tensor.shape
    image = Image.fromarray(np.squeeze(tensor))
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class MiniValidator(KC.Callback):
    def __init__(self, val_set, config, log_dir, stage, exe_rate=20):
        super().__init__()
        self.img_list = make_img_list(val_set)
        self.exe_rate = exe_rate
        self.output_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
        self.mean_pixel = config.MEAN_PIXEL
        self.log_dir = log_dir
        self.stage = stage
    
    def on_batch_end(self, batch, logs={}):
        def get_max(mask):
            if mask.dtype == np.uint8:
                return 255.0
            elif mask.dtype == np.uint16:
                return 65535.0
            else:
                raise "Unknown image format"
        def load_image(image_path):
            image = skimage.io.imread(image_path)
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            if image.shape[-1] == 4:
                image = image[..., :3]
            return image
        def load_mask(mask_path):
            mask = skimage.io.imread(mask_path)
            mask = mask.astype(np.float32) / get_max(mask)
            if mask.ndim == 3:
                mask = skimage.color.rgb2gray(mask)     # 0~1
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]
            return mask
            
        if batch % self.exe_rate != 0:
            return
        if self.stage == 'classifier':
            output_name = ['FG', 'BG', 'FG_8', 'FG_4', 'FG_2', \
                           'BG_8', 'BG_4', 'BG_2', '1-matting_GT', 'FG_GT', 'BG_GT', 'ce_weight']
        else:
            output_name = ['FG', 'BG', 'matting_result', 'fusion_mask', '1-GT']
        summary_value = []
        
        # Randomly pick a picture
        no = random.randint(1, len(self.img_list)) - 1
        img_path = self.img_list[no]

        tri_path = img_path[2]
        gt_path = img_path[1]
        img_path = img_path[0]
        
        image = load_image(img_path)
        mask = load_mask(gt_path)
        '''
        image = skimage.transform.resize(image=image, output_shape=self.output_shape,
                                        preserve_range=True, mode='reflect', anti_aliasing=True)
        fg = skimage.transform.resize(image=fg, output_shape=self.output_shape,
                                     preserve_range=True, mode='reflect', anti_aliasing=True)
        bg = skimage.transform.resize(image=bg, output_shape=self.output_shape,
                                     preserve_range=True, mode='reflect', anti_aliasing=True)
        trimap = skimage.transform.resize(image=trimap, output_shape=self.output_shape,
                                        preserve_range=True, mode='reflect', anti_aliasing=False)
        mask = skimage.transform.resize(image=mask, output_shape=self.output_shape,
                                        preserve_range=True, mode='reflect', anti_aliasing=False)
        '''
        molded_image = [image - self.mean_pixel]
        molded_mask = [mask]
        # Pack into arrays
        molded_images = np.stack(molded_image)
        molded_masks = np.stack(molded_mask)
        
        model = self.model.inner_model if hasattr(self.model,'inner_model') else self.model
        output_list = model.predict([molded_images, molded_masks], verbose=0)
        
        for j in range(0, len(output_name)):
            summary_image = make_image(np.uint8(output_list[j]*255))
            summary_value.append(tf.Summary.Value(
                tag='MiniValid_'+output_name[j], image=summary_image))
                    
        summary = tf.Summary(value=summary_value)
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summary, batch)
        writer.close()

        return

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True, anti_aliasing=False)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop

def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def _load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    # Returns
        data: Attributes data.
    """
    if name in group.attrs:
        data = [n.decode('utf8') for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while ('%s%d' % (name, chunk_id)) in group.attrs:
            data.extend([n.decode('utf8')
                        for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data

def load_weights_from_hdf5_group_by_name(f, layers, partial_loading=False, verbose=0):
    """Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = _load_attributes_from_hdf5_group(f, 'layer_names')

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    loaded_layer = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]
        # extra channel for conv1 and bn_data
        if partial_loading:
            if name == 'conv1/conv' and index[name][0].weights[0].shape[-1] != 64:
                add_channel = index[name][0].weights[0].shape[-1] - 64
                _zeros = np.zeros([(*index[name][0].weights[0].shape[:-1]), add_channel])
                weight_values[0] = np.concatenate([
                        np.array(weight_values[0]), _zeros], axis=-1)
            elif name == 'conv2_block1_1_conv' and index[name][0].weights[0].shape[-2] != 64:
                add_channel = index[name][0].weights[0].shape[-2] - 64
                _zeros = np.zeros([(*index[name][0].weights[0].shape[:-2]), \
                            add_channel, index[name][0].weights[0].shape[-1]])
                weight_values[0] = np.concatenate([
                        np.array(weight_values[0]), _zeros], axis=-2)

            elif (name == 'conv1/bn' or name == 'conv2_block1_0_bn') \
                  and index[name][0].weights[0].shape[-1] != 64:
                add_channel = index[name][0].weights[0].shape[-1] - 64
                _zeros = np.zeros(add_channel)
                _ones = np.ones(add_channel)
                weight_values[0] = np.concatenate([weight_values[0], _zeros])
                weight_values[1] = np.concatenate([weight_values[1], _ones])
                weight_values[2] = np.concatenate([weight_values[2], _zeros])
                weight_values[3] = np.concatenate([weight_values[3], _ones])

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend)
            if len(weight_values) != len(symbolic_weights):
                raise ValueError('Layer #' + str(k) +
                                 ' (named "' + layer.name +
                                 '") expects ' +
                                 str(len(symbolic_weights)) +
                                 ' weight(s), but the saved weights' +
                                 ' have ' + str(len(weight_values)) +
                                 ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i],
                                            weight_values[i]))
            if len(weight_values) != 0:
                loaded_layer.append(name)            

    # for debugging purpose
    if verbose > 0:
        print (weight_value_tuples)

    K.batch_set_value(weight_value_tuples)
    return loaded_layer

#####################################################

def save_feature_maps(features, output_dir, name, normalize=True):
    # features: batch_sizhe, h, w, c

    channel = features.shape[-1]    # channel last

    save_dir = os.path.join(output_dir, "save_features")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for c in range(channel):
        feature = features[0, :, :, c]

        if normalize:
            min_val = np.min(feature)
            max_val = np.max(feature)

            feature = (feature-min_val)*255.0 / (max_val - min_val)
            feature = np.clip(feature, 0, 255)

            skimage.io.imsave(os.path.join(save_dir, "%04d.jpg"%c), np.uint8(feature))
        else:
            skimage.io.imsave(os.path.join(save_dir, "%04d.jpg"%c), np.uint8(feature*255))

def save_feature_txt(features, name):
    channel = features.shape[-1]
    save_dir = "save_features"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for c in range(channel):
        feature = features[0, :, :, c]
        np.savetxt(os.path.join(save_dir, "%04d.txt"%c), feature)

def generate_perfect_blending_weight(FG, BG, trimap):
    FG1 = np.squeeze(FG)
    FG2 = 1.0 - np.squeeze(BG)
    trimap = np.squeeze(trimap)

    blending_weight = np.random.rand(FG.shape[0], FG.shape[1])
    visualize_blending_weight = np.random.randint(0, 255, size=(FG.shape[0], FG.shape[1], 3))

    height = FG.shape[0]
    width = FG.shape[1]

    for h in range(height):
        for w in range(width):
            if trimap[h, w] > 0.8:
                # definitely foreground
                if FG1[h, w] > FG2[h, w]:
                    # choose the larger FG1
                    blending_weight[h, w] = 1.0
                    visualize_blending_weight[h, w] = [255, 0, 0]
                elif FG1[h, w] < FG2[h, w]:
                    # choose the smaller FG2
                    blending_weight[h, w] = 0.0
                    visualize_blending_weight[h, w] = [0, 255, 0]
                # else random
            elif trimap[h, w] < 0.2:
                # definitely background
                if FG1[h, w] > FG2[h, w]:
                    blending_weight[h, w] = 0.0
                    visualize_blending_weight[h, w] = [0, 0, 255]
                elif FG1[h, w] < FG2[h, w]:
                    blending_weight[h, w] = 1.0
                    visualize_blending_weight[h, w] = [0, 255, 255]
                # else random
            # else:
            # only God knows.
    # blending_weight = blending_weight * np.squeeze(heatmap)
    # visualize_blending_weight[:, :, 0] = visualize_blending_weight[:, :, 0] * np.squeeze(heatmap)
    # visualize_blending_weight[:, :, 1] = visualize_blending_weight[:, :, 1] * np.squeeze(heatmap)
    # visualize_blending_weight[:, :, 2] = visualize_blending_weight[:, :, 2] * np.squeeze(heatmap)
    return blending_weight, np.uint8(visualize_blending_weight)