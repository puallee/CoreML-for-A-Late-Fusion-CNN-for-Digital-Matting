import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.utils import plot_model, conv_utils
import cv2
import keras2onnx
import onnx
import coremltools
from keras.utils.generic_utils import CustomObjectScope
import util 

#from bbox_image_exp import BboxImageDataset, bbox_image_data_generator
from matting_generator import MattingDataset, matting_data_generator
from loss import crossentropy_loss, l1_l2_loss, segmentation_loss, \
                 alpha_gradient_loss, alpha_sparsity_regular_graph, matting_weighted_l1_loss

from util import resize_image, log, mold_image, \
                load_weights_from_hdf5_group_by_name, MiniValidator, \
                save_feature_maps, make_img_list

from clr_callback import CyclicLR

ROOT_DIR = os.path.abspath(".")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

TRAINING_DATASET_SIZE = 147432

############################################################
#  Import network blocks
############################################################

from densenet_blocks import *
from fusion_blocks import *

############################################################
#  Config Class
############################################################
from config import Config

class ClassifierConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "classifier"

    BACKBONE = "densenet201"

    # Whether we update the variance and mean in backbone's BN layer
    BACKBONE_TRAIN_BN = None
    # Same as above but in decoder
    CLASSIFIER_TRAIN_BN = None
    
    # Whether we freeze backbone
    # True:  Use when freezing backbone (absolutely freeze everything)
    # None:  Use when training from scratch (train everything)
    # False: Use when only freezing backbone's BN layer (scale & bias)
    FREEZE_BACKBONE = None
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    GPU_COUNT = 1

    FPN_D = 128

    MEAN_PIXEL = np.array([127.156207,115.917443,106.031127])

    BINARIZE_THRESHOLD = 0.05
    
    STEPS_PER_EPOCH = TRAINING_DATASET_SIZE // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = 240

    # Don't use mini-mask since we're training a FG/BG classifier
    USE_MINI_MASK = False
    
    LEARNING_RATE = 5e-4
    MAX_LEARNING_RATE = 1.5e-3
    
    WEIGHT_DECAY = 1e-2
    
    IMAGE_SHAPE = np.array([512, 512, 3])

class FusionConfig(ClassifierConfig):
    """Configuration for fusion network / joint training.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fusion"

    FREEZE_BACKBONE = True
    BACKBONE_TRAIN_BN = False
    CLASSIFIER_TRAIN_BN = False
    FUSION_TRAIN_BN = None

    VALIDATION_STEPS = 35 

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    GPU_COUNT = 1
    STEPS_PER_EPOCH = TRAINING_DATASET_SIZE // (IMAGES_PER_GPU * GPU_COUNT)
    LEARNING_RATE = 1e-5
    MAX_LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    
    IMAGE_SHAPE = np.array([512, 512, 3])

############################################################
#  Classifier Network
############################################################

class ClassifierNetwork():

    def __init__(self, mode, config, model_dir, stage):
            """
            mode: Either "training" or "inference"
            config: A Sub-class of the Config class
            model_dir: Directory to save training logs and trained weights
            """
            assert mode in ['training', 'inference']
            self.mode = mode
            self.config = config
            self.model_dir = model_dir
            self.set_log_dir()
            self.stage = stage
            self.keras_model = self.build(mode=mode, config=config)
        
    def build(self, mode, config):
        def get_gt(input, threshold):
            raise
            x = tf.where(tf.greater(input, threshold), tf.ones_like(input), tf.zeros_like(input))
            return x
            
        """Build architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']
        
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**5 != int(h / 2**5) or w / 2**5 != int(w / 2**5):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        # We'll use fixed shape here since everything is resized / padded to
        # 512px * 512px
        #print(config.IMAGE_SHAPE)
        image_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3]
        mask_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 1]

        if mode == "training":
            raise
            input_fg_alpha_gt = KL.Input(shape=mask_shape, name='input_alpha_gt')
            input_bg_alpha_gt = KL.Lambda(lambda x: 1.0 - x)(input_fg_alpha_gt)
            input_fg_gt = KL.Lambda(lambda x: get_gt(x, config.BINARIZE_THRESHOLD))(input_fg_alpha_gt)
            input_bg_gt = KL.Lambda(lambda x: get_gt(x, config.BINARIZE_THRESHOLD))(input_bg_alpha_gt)
            ce_weight = KL.Lambda(lambda x: tf.cast(tf.logical_or(x <= config.BINARIZE_THRESHOLD, x >= 1.0 - config.BINARIZE_THRESHOLD), tf.float32))(input_fg_alpha_gt)
        
        # Although we don't need input_trimap and input_trimask during 
        # training lassifier, we keep it here for the compability of
        # the dataset generator
        input_image = KL.Input(shape=image_shape, name="input_image")
        # print(input_image)
        # raise
        # Build the shared convolutional layers.
        if config.BACKBONE[-3:] == '169':
            dense_stages = [6, 12, 32, 32]
        elif config.BACKBONE[-3:] == '201':
            dense_stages = [6, 12, 48, 32]
        else:
            dense_stages = [6, 12, 24, 16]
        skip_connections, bottleneck = \
                                DenseNet_encoder(dense_stages, input_image,
                                train_bn=config.CLASSIFIER_TRAIN_BN)
        
        FG, FG_2, FG_4, FG_8, FG_feature = \
                                DenseNet_decoder(bottleneck, skip_connections,
                                network_name="deFG", train_bn=config.CLASSIFIER_TRAIN_BN,
                                fpn_d=config.FPN_D, output_dim=mask_shape[0:2],)

        BG, BG_2, BG_4, BG_8, BG_feature = \
                                DenseNet_decoder(bottleneck, skip_connections,
                                network_name="deBG", train_bn=config.CLASSIFIER_TRAIN_BN,
                                fpn_d=config.FPN_D, output_dim=mask_shape[0:2],)

        if mode == "training":
            # Training graph
            if self.stage == 'classifier':
                # Classifier training losses
                FG_mask_ce_loss = KL.Lambda(lambda x: crossentropy_loss(*x), \
                                        name="FG_mask_ce_loss")([FG, input_fg_gt, input_fg_alpha_gt, ce_weight])
                FG_half_mask_ce_loss = KL.Lambda(lambda x: crossentropy_loss(*x), \
                                        name="FG_half_ce_loss")([FG_2, input_fg_gt, input_fg_alpha_gt, ce_weight])
                FG_fourth_mask_ce_loss = KL.Lambda(lambda x: crossentropy_loss(*x), \
                                        name="FG_fourth_ce_loss")([FG_4, input_fg_gt, input_fg_alpha_gt, ce_weight])
                FG_eighth_mask_ce_loss = KL.Lambda(lambda x: crossentropy_loss(*x), \
                                        name="FG_eighth_ce_loss")([FG_8, input_fg_gt, input_fg_alpha_gt, ce_weight])
                
                BG_mask_ce_loss = KL.Lambda(lambda x: crossentropy_loss(*x), \
                                        name="BG_mask_ce_loss")([BG, input_bg_gt, input_bg_alpha_gt, ce_weight])
                BG_half_mask_ce_loss = KL.Lambda(lambda x: crossentropy_loss(*x), \
                                        name="BG_half_ce_loss")([BG_2, input_bg_gt, input_bg_alpha_gt, ce_weight])
                BG_fourth_mask_ce_loss = KL.Lambda(lambda x: crossentropy_loss(*x), \
                                        name="BG_fourth_ce_loss")([BG_4, input_bg_gt, input_bg_alpha_gt, ce_weight])
                BG_eighth_mask_ce_loss = KL.Lambda(lambda x: crossentropy_loss(*x), \
                                        name="BG_eighth_ce_loss")([BG_8, input_bg_gt, input_bg_alpha_gt, ce_weight])
                                        
                FG_mask_l_loss = KL.Lambda(lambda x: l1_l2_loss(*x), \
                                        name="FG_mask_l_loss")([FG, input_fg_gt, input_fg_alpha_gt, ce_weight])
                FG_half_mask_l_loss = KL.Lambda(lambda x: l1_l2_loss(*x), \
                                        name="FG_half_l_loss")([FG_2, input_fg_gt, input_fg_alpha_gt, ce_weight])
                FG_fourth_mask_l_loss = KL.Lambda(lambda x: l1_l2_loss(*x), \
                                        name="FG_fourth_l_loss")([FG_4, input_fg_gt, input_fg_alpha_gt, ce_weight])
                FG_eighth_mask_l_loss = KL.Lambda(lambda x: l1_l2_loss(*x), \
                                        name="FG_eighth_l_loss")([FG_8, input_fg_gt, input_fg_alpha_gt, ce_weight])
                
                BG_mask_l_loss = KL.Lambda(lambda x: l1_l2_loss(*x), \
                                        name="BG_mask_l_loss")([BG, input_bg_gt, input_bg_alpha_gt, ce_weight])
                BG_half_mask_l_loss = KL.Lambda(lambda x: l1_l2_loss(*x), \
                                        name="BG_half_l_loss")([BG_2, input_bg_gt, input_bg_alpha_gt, ce_weight])
                BG_fourth_mask_l_loss = KL.Lambda(lambda x: l1_l2_loss(*x), \
                                        name="BG_fourth_l_loss")([BG_4, input_bg_gt, input_bg_alpha_gt, ce_weight])
                BG_eighth_mask_l_loss = KL.Lambda(lambda x: l1_l2_loss(*x), \
                                        name="BG_eighth_l_loss")([BG_8, input_bg_gt, input_bg_alpha_gt, ce_weight])

                
                FG_mask_alpha_gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), \
                                        name="FG_mask_gradient_loss")([FG, input_fg_alpha_gt])
                FG_half_mask_alpha_gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), \
                                        name="FG_half_gradient_loss")([FG_2, input_fg_alpha_gt])
                FG_fourth_mask_alpha_gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), \
                                        name="FG_fourth_gradient_loss")([FG_4, input_fg_alpha_gt])
                FG_eighth_mask_alpha_gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), \
                                        name="FG_eighth_gradient_loss")([FG_8, input_fg_alpha_gt])
                
                BG_mask_alpha_gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), \
                                        name="BG_mask_gradient_loss")([BG, input_bg_alpha_gt])
                BG_half_mask_alpha_gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), \
                                        name="BG_half_gradient_loss")([BG_2, input_bg_alpha_gt])
                BG_fourth_mask_alpha_gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), \
                                        name="BG_fourth_gradient_loss")([BG_4, input_bg_alpha_gt])
                BG_eighth_mask_alpha_gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), \
                                        name="BG_eighth_gradient_loss")([BG_8, input_bg_alpha_gt])
                # Models
                inputs = [input_image, input_fg_alpha_gt]
                outputs = [FG, BG, FG_8, FG_4, FG_2, BG_8, BG_4, BG_2, \
                        input_bg_alpha_gt, input_fg_gt, input_bg_gt, ce_weight,
                        FG_mask_ce_loss, FG_half_mask_ce_loss, FG_fourth_mask_ce_loss, FG_eighth_mask_ce_loss,\
                        BG_mask_ce_loss, BG_half_mask_ce_loss, BG_fourth_mask_ce_loss, BG_eighth_mask_ce_loss,\
                        FG_mask_l_loss, FG_half_mask_l_loss, FG_fourth_mask_l_loss, FG_eighth_mask_l_loss,\
                        BG_mask_l_loss, BG_half_mask_l_loss, BG_fourth_mask_l_loss, BG_eighth_mask_l_loss,\
                        FG_mask_alpha_gradient_loss, FG_half_mask_alpha_gradient_loss, FG_fourth_mask_alpha_gradient_loss, FG_eighth_mask_alpha_gradient_loss,\
                        BG_mask_alpha_gradient_loss, BG_half_mask_alpha_gradient_loss, BG_fourth_mask_alpha_gradient_loss, BG_eighth_mask_alpha_gradient_loss]
                model = KM.Model(inputs, outputs, name='classifier_network')

            elif self.stage == 'fusion':
                # Fusion blocks & training losses

                fusion_mask, fusion_features = fusion_graph_with_rgb(fg_features=FG_feature, bg_features=BG_feature,
                                                                     filters=[256, 128, 128, 64, 256], input_rgb=input_image,
                                                                     train_bn=config.FUSION_TRAIN_BN, d=config.FPN_D)

                matting_result = blending_graph(FG, BG, fusion_mask)

                matting_l1_loss = KL.Lambda(lambda x: matting_weighted_l1_loss(*x), name="matting_l1")(
                    [input_fg_alpha_gt, matting_result, 1. - ce_weight])
                sparsity_loss = KL.Lambda(lambda x: 0.01 * alpha_sparsity_regular_graph(x), name="sparsity_loss")(
                    matting_result)
                gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), name="matting_gradient_loss")(
                    [matting_result, input_fg_alpha_gt])

                # Models
                inputs = [input_image, input_fg_alpha_gt]
                outputs = [FG, BG, matting_result, fusion_mask, input_bg_alpha_gt, \
                           fusion_features[0], fusion_features[1],
                           matting_l1_loss, sparsity_loss, gradient_loss]
                model = KM.Model(inputs, outputs, name='fusion_network')
            else:  # 'joint'
                # Fusion blocks & training losses

                fusion_mask, fusion_features = fusion_graph_with_rgb(fg_features=FG_feature, bg_features=BG_feature,
                                                                     filters=[256, 128, 128, 64, 256], input_rgb=input_image,
                                                                     train_bn=config.FUSION_TRAIN_BN, d=config.FPN_D)

                matting_result = blending_graph(FG, BG, fusion_mask)

                FG_mask_loss = KL.Lambda(lambda x: 0.5 * segmentation_loss(*x), \
                                         name="FG_mask_loss")([FG, input_fg_gt, input_fg_alpha_gt, ce_weight])
                BG_mask_loss = KL.Lambda(lambda x: 0.5 * segmentation_loss(*x), \
                                         name="BG_mask_loss")([BG, input_bg_gt, input_bg_alpha_gt, ce_weight])
                FG_half_loss = KL.Lambda(lambda x: 0.5 * segmentation_loss(*x), \
                                         name='FG_half_loss')([FG_2, input_fg_gt, input_fg_alpha_gt, ce_weight])
                BG_half_loss = KL.Lambda(lambda x: 0.5 * segmentation_loss(*x), \
                                         name='BG_half_loss')([BG_2, input_bg_gt, input_bg_alpha_gt, ce_weight])

                FG_fourth_loss = KL.Lambda(lambda x: 0.5 * segmentation_loss(*x), \
                                           name='FG_fourth_loss')([FG_4, input_fg_gt, input_fg_alpha_gt, ce_weight])
                BG_fourth_loss = KL.Lambda(lambda x: 0.5 * segmentation_loss(*x), \
                                           name='BG_fourth_loss')([BG_4, input_bg_gt, input_bg_alpha_gt, ce_weight])

                FG_eighth_loss = KL.Lambda(lambda x: 0.5 * segmentation_loss(*x), \
                                           name='FG_eighth_loss')([FG_8, input_fg_gt, input_fg_alpha_gt, ce_weight])
                BG_eighth_loss = KL.Lambda(lambda x: 0.5 * segmentation_loss(*x), \
                                           name='BG_eighth_loss')([BG_8, input_bg_gt, input_bg_alpha_gt, ce_weight])

                matting_l1_loss = KL.Lambda(lambda x: matting_weighted_l1_loss(*x), name="matting_l1")(
                    [input_fg_alpha_gt, matting_result, 1. - ce_weight])
                sparsity_loss = KL.Lambda(lambda x: 0.01 * alpha_sparsity_regular_graph(x), name="sparsity_loss")(
                    matting_result)
                gradient_loss = KL.Lambda(lambda x: alpha_gradient_loss(*x), name="matting_gradient_loss")(
                    [matting_result, input_fg_alpha_gt])

                # Models
                inputs = [input_image, input_fg_alpha_gt]
                outputs = [FG, BG, matting_result, fusion_mask, input_bg_alpha_gt, \
                           fusion_features[0], fusion_features[1], matting_l1_loss, \
                           FG_mask_loss, BG_mask_loss, FG_half_loss, BG_half_loss, \
                           FG_fourth_loss, BG_fourth_loss, FG_eighth_loss, BG_eighth_loss, \
                           sparsity_loss, gradient_loss]
                model = KM.Model(inputs, outputs, name='fusion_network')
        else:
            # Inference graph
            if self.stage == 'classifier':
                # Classifier inference
                DCFG = KL.Activation('sigmoid')(FG_feature)
                DCBG = KL.Activation('sigmoid')(BG_feature)
                model = KM.Model([input_image], [FG, BG, \
                            FG_8, FG_4, FG_2, \
                            BG_8, BG_4, BG_2, DCFG, DCBG, \
                            ], name='classifier_network')
            else:
                # Fusion inference
                ## lambda have: generate_trimap
                # [fake_trimap, fake_trimask] = KL.Lambda(generate_trimap, name='erosion_layer', \
                #                                         arguments={'edge_width': 20})([FG, BG])

                fusion_mask, fusion_features = fusion_graph_with_rgb(fg_features=FG_feature, bg_features=BG_feature,
                                                                     filters=[256, 128, 128, 64, 256], input_rgb=input_image,
                                                                     train_bn=config.FUSION_TRAIN_BN, d=config.FPN_D)

                matting_result = blending_graph(FG, BG, fusion_mask)
                # remove lambda
                # _one = KL.Lambda(lambda x: tf.ones_like(x))(fake_trimask)
                # _zero = KL.Lambda(lambda x: tf.zeros_like(x))(fake_trimask)

               # inputs = [input_image]
                inputs=input_image
              #  outputs = [FG, BG, matting_result, fusion_mask, fake_trimap, fake_trimask]
                outputs=matting_result
                model = KM.Model(inputs, outputs, name='fusion_network')
        
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            print('true')
            raise
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)#, use_gpu=[0, 3])
       
        return model
    
    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """

        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint
     
    def load_weights(self, filepath, by_name=False, restart=False,
                     exclude=None, verbose=0):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            loaded_layers = load_weights_from_hdf5_group_by_name(f, layers)
            if verbose > 0:
                log(loaded_layers)
                log(str(len(loaded_layers)) + " layers loaded.")
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        if not restart:
            self.set_log_dir(filepath)

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Add Losses
        # First, clear previously set losses to avoid duplication
        
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = []
        if self.stage == 'classifier':
            loss_names = [
                "FG_mask_ce_loss",
                "FG_half_ce_loss",
                "FG_fourth_ce_loss",
                "FG_eighth_ce_loss",
                "BG_mask_ce_loss",
                "BG_half_ce_loss",
                "BG_fourth_ce_loss",
                "BG_eighth_ce_loss",
                "FG_mask_l_loss",
                "FG_half_l_loss",
                "FG_fourth_l_loss",
                "FG_eighth_l_loss",
                "BG_mask_l_loss",
                "BG_half_l_loss",
                "BG_fourth_l_loss",
                "BG_eighth_l_loss",
                "FG_mask_gradient_loss",
                "FG_half_gradient_loss",
                "FG_fourth_gradient_loss",
                "FG_eighth_gradient_loss",
                "BG_mask_gradient_loss",
                "BG_half_gradient_loss",
                "BG_fourth_gradient_loss",
                "BG_eighth_gradient_loss"]
        elif self.stage == 'fusion':
            loss_names = ["matting_l1", "sparsity_loss", "matting_gradient_loss"]
        elif self.stage == 'joint':
            loss_names = ["matting_l1",
                          "FG_mask_loss",
                          "BG_mask_loss",
                          "FG_half_loss",
                          "BG_half_loss",
                          "FG_fourth_loss",
                          "BG_fourth_loss",
                          "FG_eighth_loss",
                          "BG_eighth_loss",
                          "matting_gradient_loss",
                          "sparsity_loss"]

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            #loss = tf.reduce_mean(layer.output, keep_dims=True)
            self.keras_model.add_loss(layer.output)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        # Not needed for now
        reg_losses = []
        for w in self.keras_model.trainable_weights:
            if 'bn' not in w.name:
                # is conv layer
                if 'conv1/conv' in w.name or 'deconv1_input' in w.name:
                    reg_losses.append(keras.regularizers.l1(0.01)(w) / tf.cast(tf.size(w), tf.float32))
                else:
                    reg_losses.append(keras.regularizers.l2(0.01)(w) / tf.cast(tf.size(w), tf.float32))
        reg_losses_sum = tf.add_n(reg_losses)
        self.keras_model.add_loss(reg_losses_sum)
        
        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))
        
        # Add metrics for losses
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if name in self.keras_model.metrics_names:
                continue
            self.keras_model.metrics_names.append(name)
            #loss = (
                #tf.reduce_mean(layer.output, keep_dims=True)
                #* self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(layer.output)
        self.keras_model.metrics_names.append('L2_reg_loss')
        self.keras_model.metrics_tensors.append(reg_losses_sum)
        return self.keras_model

    def set_trainable(self, layer_regex, keras_model=None, exclude_mode=False,
                      fullmatch=False, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        def print_train_layers(model):
            train_layers_name = []
            for layer in model.get_training_layers():
                train_layers_name.append(layer.name)
            print(train_layers_name)
            
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            if fullmatch:
                trainable = bool(re.fullmatch(layer_regex, layer.name))
            else:
                trainable = bool(re.search(layer_regex, layer.name))
            if exclude_mode:
                trainable = not trainable
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

        # Extra verbose: Print training layers,
        # make sure encoder is freezed
        if verbose > 1: print_train_layers(self)

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/classifier\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "classifier_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, epochs, data_root):
        assert self.mode == "training", "Create model in training mode."

        # Data preparation
        train_dataset = MattingDataset(data_root=data_root, subset='train')
        val_dataset = MattingDataset(data_root=data_root, subset='val')
        train_generator = matting_data_generator(train_dataset, self.config, shuffle=True,
                                                  augment=True, batch_size=self.config.BATCH_SIZE)
        val_generator = matting_data_generator(val_dataset, self.config, shuffle=False,
                                               batch_size=self.config.BATCH_SIZE,
                                               trunc=self.config.BATCH_SIZE*self.config.VALIDATION_STEPS)
        
        # Callbacks
        callbacks = [
            CyclicLR(base_lr=self.config.LEARNING_RATE,
                     max_lr=self.config.MAX_LEARNING_RATE,
                     momentum=self.config.LEARNING_MOMENTUM,
                     step_size=self.config.STEPS_PER_EPOCH * 2,
                     resume_iter=self.epoch*self.config.STEPS_PER_EPOCH),
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=False, write_images=False),
            MiniValidator(log_dir=self.log_dir, stage=self.stage,
			   val_set=os.path.join(val_dataset._data_root,'val_set.txt'), \
                           config=self.config, exe_rate=100),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=1, save_weights_only=False),
        ]

        # Train
        print("\nStarting at epoch {}. lr={}\n".format(self.epoch, self.config.LEARNING_RATE))
        print("checkpoint path: {}".format(self.checkpoint_path))

        # Set trainable layers
        exclude = False
        if self.config.FREEZE_BACKBONE is None:
            layers = '.*'               # train from scratch
        elif self.config.FREEZE_BACKBONE == True:
            if self.stage == 'fusion':       # fusion training
                layers = '^(fusion)'
            else:
                layers = '^(deFG)|(deBG)' 
        else:
            if self.stage == 'fusion' or self.stage == 'joint':       # joint train
                layers = 'bn'
            if self.stage == 'classifier':   # classifier training
                layers = '^(bn)'
            exclude = True
        self.set_trainable(layers, exclude_mode=exclude, verbose=1)

        self.compile(self.config.LEARNING_RATE, self.config.LEARNING_MOMENTUM)

        # We don't use multiprocessing or any threads here,
        # since it might get hanged due to the fact that the
        # data generator is probably not multi-thread safe.
        workers = 0

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=False,
            verbose=1)
        self.epoch = max(self.epoch, epochs)
    
    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images = []
        for i in range(len(images)):
            image = images[i].astype(np.float32)
            #print(config.IMAGE_SHAPE[0:2])
            s=config.IMAGE_SHAPE[0:2]
            image = skimage.transform.resize(image=image, output_shape=s,
                        preserve_range=True, mode='reflect', anti_aliasing=True)
            molded_image = image - config.MEAN_PIXEL
            
            molded_images.append(molded_image)

        # Pack into arrays
        molded_images = np.stack(molded_images)
        # print(np.array(image)- config.MEAN_PIXEL)
        print(config)
        # print(molded_images)
        # raise
        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        if verbose:
            log("molded_images", molded_images)
            
        # Run object detection
        output_list =\
            self.keras_model.predict([molded_images], verbose=0)
        # Process detections
        return output_list,self.keras_model
        
    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def get_training_layers(self):
        """Returns a list of layers that is not freezed whiled training."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.trainable and l.get_weights():
                layers.append(l)
        return layers

############################################################
#  Routines for inference
############################################################

def infer_classifier_dataset(model, output_dir, data_root, config, test_num=None, verbose=0, subset='val', output=True):        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = MattingDataset(data_root, subset=subset)
    generator = matting_data_generator(dataset, config, False, False, config.BATCH_SIZE)
    s = ''
    id = 0
    avg_loss = avg_fg_loss = avg_bg_loss = avg_fgbg_loss = 0
    improved_count = 0
    if not test_num:
        test_num = len(dataset.image_ids)
    output_name = ['FG', 'BG', 'FG_8', 'FG_4', 'FG_2', \
                    'BG_8', 'BG_4', 'BG_2']
    sum_fg_loss = sum_bg_loss = 0
    sum_fg_l1 = sum_bg_l1 = 0
    for inputs, _ in generator:
        id += 1
        if id > test_num:
            break

        batch_images = inputs[0]    # molded
        batch_trimaps = inputs[1]   # 0~1
        batch_trimasks = inputs[2]  # 0~1
        gt = inputs[-1]

        output_list = \
            model.predict(inputs)
        for i in range(0, len(output_list)):
            output_list[i] = np.squeeze(output_list[i])
        fg_ce_losses = output_list[-24:-20]
        bg_ce_losses = output_list[-20:-16]
        fg_l1_losses = output_list[-16:-12]
        bg_l1_losses = output_list[-12:-8]
        fg_grad_l1_losses = output_list[-8:-4]
        bg_grad_l1_losses = output_list[-4:]
        heat = np.abs(output_list[0]+output_list[1]-1)
        heat_min = np.min(heat)
        heat_max = np.max(heat)
        heat = np.uint8((heat - heat_min) / (heat_max - heat_min) * 255.0)
        
        imgname = os.path.join(output_dir, "%06d"%id)
        print(inputs[0].shape)
        if output:
            for i in range(0, len(output_name)):
                skimage.io.imsave(imgname + '_' + output_name[i] + '.png', output_list[i])
            skimage.io.imsave(imgname + '_rgb.jpg', np.uint8(np.squeeze(batch_images)+config.MEAN_PIXEL))
            skimage.io.imsave(imgname + '_heat.png', heat)
        
        fg_errors = fg_ce_losses
        fg_errors.extend(fg_l1_losses)
        fg_errors.extend(fg_grad_l1_losses)
        bg_errors = bg_ce_losses
        bg_errors.extend(bg_l1_losses)
        bg_errors.extend(bg_grad_l1_losses)
        fg_sums = np.sum(fg_errors)
        bg_sums = np.sum(bg_errors)
        sum_fg_loss += fg_sums
        sum_bg_loss += bg_sums
        
        gt = np.squeeze(gt)
        fg_l1 = np.mean(np.abs(output_list[0]-gt))
        bg_l1 = np.mean(np.abs(1-output_list[1]-gt))
        sum_fg_l1 += fg_l1
        sum_bg_l1 += bg_l1

        _print = 'Infer: %d / %d.\n'\
                % (id, test_num)
        _print += 'loss=%.4f FG_loss=%.4f BG_loss=%.4f\n' % (fg_sums+bg_sums, fg_sums, bg_sums)
        _print += 'FG_l1=%.4f BG_l1=%.4f' % (fg_l1, bg_l1)
        print (_print)
        s += (_print+'\n')

    _print = " \n"
    _print += "Average Fusion Loss: %.4f\n" % ((sum_fg_loss+sum_bg_loss) / test_num)
    _print += "Average Foreground Classifier Loss: %.4f\n" % (sum_fg_loss / test_num)
    _print += "Average Background Classifier Loss: %.4f\n" % (sum_bg_loss / test_num)
    _print += "Average FG L1 Error: %.4f\n" % (sum_fg_l1 / test_num)
    _print += "Average BG L1 Error: %.4f\n" % (sum_bg_l1 / test_num)
    
    print (_print)
    s += _print
    f=open(os.path.join(output_dir, os.path.basename(output_dir)+'.txt'), 'w')
    f.write(s)
    f.close()

def infer_classifier(model, img_lists, output_dir=None):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_count = len(img_lists)
    output_name = ['FG', 'BG', 'FG_8', 'FG_4', 'FG_2', \
                    'BG_8', 'BG_4', 'BG_2']
    for i, img_path in enumerate(img_lists):
        img_path = img_path[0]
        print ('Infer: %d / %d' % (i, img_count), end="\r")
        # Read image first
        image = skimage.io.imread(img_path)
        # if has an alpha channel, remove it
        if image.shape[-1] == 4:
                image = image[..., :3]
        # if grayscale, convert to rgb
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        
        output_list,net = model.detect([image])
        net.save('tpp.h5')
        for i in range(0, len(output_list)):
            output_list[i] = np.squeeze(output_list[i])
        heat = np.abs(output_list[0]+output_list[1]-1)
        heat_min = np.min(heat)
        heat_max = np.max(heat)
        heat = np.uint8((heat - heat_min) / (heat_max - heat_min) * 255.0)

        if output_dir:
            imgname = os.path.join(output_dir, os.path.basename(img_path))[:-4]
        else:
            imgname = img_path[:-4]
        for i in range(0, len(output_name)):
            skimage.io.imsave(imgname + '_' + output_name[i] + '.png', output_list[i])
        skimage.io.imsave(imgname + '_heat.png', heat)
       


def infer_fusion(model, img_lists, output_dir=None, verbose=1):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_count = len(img_lists)
  #  output_name = ['FG', 'BG', 'matting', 'fusion_mask', 'fake_trimap', 'fake_trimask']
    output_name = ['FG', 'BG', 'matting', 'fusion_mask']
    for i, img_path in enumerate(img_lists):
        img_path = img_path[0]
        print ('Infer: %d / %d' % (i, img_count), end="\r")
        # Read image first
        image = skimage.io.imread(img_path)
        # if has an alpha channel, remove it
        if image.shape[-1] == 4:
                image = image[..., :3]
        # if grayscale, convert to rgb
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        
       # model.save('./allmodel.h5')
        output_list,net = model.detect([image])
        #net=ClassifierNetwork().build
        #print(dir(net))
       # os._exit()
        # net.save('tp.h5')
        # # onnx_model = keras2onnx.convert_keras(net,net.name)
        # # temp_model_file = './model.onnx'
        # # onnx.save_model(onnx_model, temp_model_file)
        with CustomObjectScope({'BatchNorm': keras.layers.BatchNormalization},{'Subone':util.Subone}
                        ):
            model = coremltools.converters.keras.convert(net,
                                  input_names=['input'],
                                  output_names='output',
                                  image_input_names='input',
                                  image_scale=1,
                                  red_bias=-127.156207,
                                  green_bias=-115.917443,
                                  blue_bias=-106.031127,
                                  model_precision='float32',
                                  use_float_arraytype=True)
            model.save('mattingRGB.mlmodel')
            print(model)
        alpha=np.squeeze(output_list)+1
        w,h=alpha.shape
        for i in range (w):
          for j in range(h):
                    if alpha[i,j]<0.82:
                        alpha[i,j]=0
        img=cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        w1,h1,c=img.shape
        out=cv2.resize(img,(h,w), interpolation=cv2.INTER_LANCZOS4)
        for i in range(3):
          out[:,:,i]=out[:,:,i]*(alpha)
        for i in range (w):
          for j in range(h):
                    if out[i,j,2]==0:
                        out[i,j,2]=255
        out=cv2.resize(out,(h1,w1), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite('./result/out5.jpg',alpha*255)
        raise
        for i in range(0, len(output_list)):
            output_list[i] = np.squeeze(output_list[i])
        heat = np.abs(output_list[0]+output_list[1]-1)
        heat_min = np.min(heat)
        heat_max = np.max(heat)
        heat = np.uint8((heat - heat_min) / (heat_max - heat_min) * 255.0)

        if output_dir:
            imgname = os.path.join(output_dir, os.path.basename(img_path))[:-4]
        else:
            imgname = img_path[:-4]
        for i in range(0, len(output_name)):
            skimage.io.imsave(imgname + '_' + output_name[i] + '.png', output_list[i])
        skimage.io.imsave(imgname + '_heat.png', heat)
        alpha=output_list[2]
        w,h=alpha.shape
        for i in range (w):
          for j in range(h):
                    if alpha[i,j]<0.62:
                        alpha[i,j]=0
        img=cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        w1,h1,c=img.shape
        out=cv2.resize(img,(h,w), interpolation=cv2.INTER_LANCZOS4)
      #  alpha=cv2.resize(alpha,(h,w), interpolation=cv2.INTER_LANCZOS4)

        
        for i in range(3):
          out[:,:,i]=out[:,:,i]*(alpha)
        for i in range (w):
          for j in range(h):
                    if out[i,j,2]==0:
                        out[i,j,2]=255
        out=cv2.resize(out,(h1,w1), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite('./result/out5.jpg',out)
       #  ma=alpha.max()
       #  mi=alpha.min()
       #  alpha=(alpha-mi)/(ma-mi)
       #  alpha=alpha*255
       #  np.set_printoptions(threshold=np.inf) 
        
       #  alpha = (alpha).astype(np.uint8)
       #  _,alpha = cv2.threshold(alpha.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
       #  print(alpha)
       #  r, g, b = cv2.split(out)
       #  bgra = [r,g,b,alpha]
       #  four = cv2.merge(bgra,4)
       # # print(outt.shape)
       #  cv2.imwrite('./result/out5.jpg',four)
       #  cv2.imwrite('./result/out5mask.jpg',four[:,:,3])

    print (" ")
    print (" ")
    
def infer_fusion_dataset(model, output_dir, data_root, config, test_num=None, verbose=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = MattingDataset(data_root, subset='val')
    generator = matting_data_generator(dataset, config, False, False, config.BATCH_SIZE)
    s = ''
    i = 0
    avg_loss = avg_fg_loss = avg_bg_loss = avg_fgbg_loss = 0
    improved_count = 0
    if not test_num:
        test_num = len(dataset.image_ids)
    for inputs, _ in generator:
        i += 1
        if i > test_num:
            break

        batch_images = inputs[0]    # molded
        batch_trimaps = inputs[1]   # 0~1
        gt = inputs[3]

        FG_out, BG_out, fusion_result, fusion_mask, features, trimask = \
            model.predict([batch_images])
        features = np.concatenate([features, trimask], axis=-1)
        FG_out = FG_out[0]
        BG_out = BG_out[0]
        heat_map = np.abs(FG_out*1.0 + BG_out*1.0 - 1.0)
        fusion_result = fusion_result[0]
        fusion_mask = fusion_mask[0]
        gt = gt[0]


        l2_fusion_error = np.mean(np.abs(gt - fusion_result))
        l2_fgbg_error = np.mean(np.abs(gt - ((FG_out+1-BG_out)/2.0)))
        l2_fg_error = np.mean(np.abs(gt - FG_out))
        l2_bg_error = np.mean(np.abs(gt - (1-BG_out)))

        avg_loss += l2_fusion_error
        avg_fgbg_loss += l2_fgbg_error
        avg_fg_loss += l2_fg_error
        avg_bg_loss += l2_bg_error
        
        show_FG = np.uint8(np.squeeze(FG_out)*255)
        show_BG = np.uint8(np.squeeze(BG_out)*255)
        show_FGBG = np.concatenate([show_FG, show_BG], axis=0)
        
        show_fusion = np.uint8(np.squeeze(fusion_result)*255)
        show_trimap = np.uint8(np.squeeze(features[0][...,0])*255)
        show_ft = np.concatenate([show_fusion, show_trimap], axis=0)
        
        show_FBFT = np.concatenate([show_ft, show_FGBG], axis=1)[..., np.newaxis]
        show_FBFT = show_FBFT.repeat(3, axis=-1)
        
        show_GT = np.uint8(np.squeeze(gt)*255)[..., np.newaxis]
        show_GT = show_GT.repeat(3, axis=-1)
        
        show_img = np.uint8(np.squeeze(inputs[0] + config.MEAN_PIXEL))
        show_imgt = np.concatenate([show_img, show_GT], axis=0)
        
        show_all = np.concatenate([show_imgt, show_FBFT], axis=1)
        skimage.io.imsave(os.path.join(output_dir, "%06d.jpg"%i), show_all)
        skimage.io.imsave(os.path.join(output_dir, "%06d_Blending.png"%i), np.uint8(np.squeeze(fusion_mask)*255))

        _print = 'Infer: %d / %d. fusion_loss=%.4f, fgbg_avg_loss=%.4f, fg_loss=%.4f, bg_loss=%.4f, improved:%d'\
                % (i, test_num, l2_fusion_error, l2_fgbg_error, l2_fg_error, l2_bg_error, l2_fusion_error < l2_fgbg_error)
        print (_print)
        s += (_print + '\n')
        if l2_fusion_error < l2_fgbg_error:
            improved_count += 1

    _print = " \n"
    _print += "Average Fusion Loss: %.4f\n" % (avg_loss / test_num)
    _print += "Average (FG+BG)/2 Loss: %.4f\n" % (avg_fgbg_loss / test_num)
    _print += "Average Foreground Classifier Loss: %.4f\n" % (avg_fg_loss / test_num)
    _print += "Average Background Classifier Loss: %.4f\n" % (avg_bg_loss / test_num)
    _print += "improved: %d/%d\n" % (improved_count, test_num)

    print (_print)
    s += _print
    f=open(os.path.join(output_dir, os.path.basename(output_dir)+'.txt'), 'w')
    f.write(s)
    f.close()

if __name__=="__main__":
    
  #  tensorflow backend config
#     tfconfig = tf.ConfigProto()
#     tfconfig=tf.compat.v1.ConfigProto()
#     tfconfig.gpu_options.allow_growth = True
#     tfconfig.allow_soft_placement = True
#     #session = tf.Session(config=tfconfig)
#    # session = tf.compat.v1.Session(config=tfconfig)
#     session=tf.compat.v1.disable_eager_execution()
#    # K.tensorflow_backend.set_session(session)
#     tf.compat.v1.keras.backend.set_session(
#     session
# )

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=2,
                        metavar="<command> <training stage>",
                        help="command: 'train' or 'infer', stage: 'classifier', 'fusion' or 'joint'.")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image or a image list",
                        help='Image to be inferenced.')
    parser.add_argument('--outdir', required=False,
                        metavar="/path/to/output/dir/",
                        help='Inference result output directory.')
    parser.add_argument('--info', required=False, action='store_true',
                        help='Print out network architecture.')
    parser.add_argument('--freeze_backbone', required=False, action='store_true',
                        help='Freeze backbone or not while training.')
    parser.add_argument('--restart', required=False, action='store_true',
                        help='Force restart training from epoch=0.')
    args = parser.parse_args()

    # Validate arguments
    command = args.command[0]
    stage = args.command[1]

    assert command in ['train', 'infer'], "Command can only be 'train' or 'infer'"
    assert stage in ['classifier', 'fusion', 'joint'], \
            "Training stage can only be 'classifier', 'fusion'."

    if command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    verbose = int(args.info)

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Verbose Info:", args.info)
    
    # Configurations
    if command == "train":
        mode = 'training'
        config = ClassifierConfig() if stage == 'classifier' else FusionConfig()

        # By default, if we load a pretrained model, we only freeze its BN layers
        # config.FREEZE_BACKBONE = False
        # config.BACKBONE_TRAIN_BN = False

        # Absolutely freeze everything in backbone if this switch is on
        if args.freeze_backbone:
            assert args.weights, \
                "You must provide a pretrain model as backbone with --weights."
            config.FREEZE_BACKBONE = True
            config.BACKBONE_TRAIN_BN = False
        else:
            if stage == 'fusion': # joint train
                config.STEPS_PER_EPOCH = TRAINING_DATASET_SIZE // (config.GPU_COUNT * config.IMAGES_PER_GPU)
                config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
                config.FREEZE_BACKBONE = True
            elif stage == 'joint':
                config.STEPS_PER_EPOCH = TRAINING_DATASET_SIZE // (config.GPU_COUNT * config.IMAGES_PER_GPU)
                config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
                config.FREEZE_BACKBONE = None
                config.FUSION_TRAIN_BN = False
                print("\njoint training\n")
                
        # Training from scratch means we have to train everything
        if not args.weights:
            config.BACKBONE_TRAIN_BN = None
            config.FREEZE_BACKBONE = None
    else:
        mode = 'inference'
        if stage == 'classifier':
            if args.dataset:
                mode = 'training'
            class InferenceConfig(ClassifierConfig):
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
                CLASSIFIER_TRAIN_BN = False
                BACKBONE_TRAIN_BN = False
                IMAGE_SHAPE = np.array([640, 960, 3])
            config = InferenceConfig()
        else:
            class InferenceConfig(FusionConfig):
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
                FUSION_TRAIN_BN = False
                BACKBONE_TRAIN_BN = False
                CLASSIFIER_TRAIN_BN = False
                IMAGE_SHAPE = np.array([640, 960, 3])
            config = InferenceConfig()
          

    if config.BACKBONE[5] == 'X':
        config.MEAN_PIXEL = np.array([0,0,0])
    config.display()
    print(mode)
    # Create model
    model = ClassifierNetwork(mode=mode, config=config, stage=stage,
                              model_dir=args.logs)

    # Load weights
    if args.weights:
        print("Loading weights ", args.weights)
        model.load_weights(args.weights, by_name=True, verbose=verbose,
                           restart=args.restart)

    # Writting debug informations
    if args.info:
        with open('classifier_arch.txt', 'w') as f:
            model.keras_model.summary(line_length=200,
                                    print_fn=lambda x: f.write(x+'\n'))

    if command == "train":
        print ("Training network...")
        model.train(epochs=100, data_root=args.dataset)
    elif command == "infer":
        if not args.image and not args.dataset:
            raise "At least one of '--dataset' or '--image' should be provided for inference."
        if stage == 'classifier':
            if args.image:
                img_list = make_img_list(args.image)
                infer_classifier(model, img_list, os.path.abspath(args.outdir))
            else:
                infer_classifier_dataset(model=model.keras_model, output_dir=args.outdir,
                    data_root=args.dataset, config=config, subset='val', output=False)
        else:
            if args.dataset:
                infer_fusion_dataset(model=model.keras_model, output_dir=args.outdir,
                    data_root=args.dataset, config=config)
            else:
                img_list = make_img_list(args.image)
                infer_fusion(model, img_list, os.path.abspath(args.outdir))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'infer'".format(command))
