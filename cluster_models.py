from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import h5py
import threading
import itertools
from collections import deque
import sys,datetime,time,os
import keras
import keras.backend as K
import tensorflow as tf
import argparse
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import custom_layers

########################################################################
###  Models Define
########################################################################

def encoder_decoder_timechunk( height, width, timesteps, convFilters, num_clusters, initial_clusters, num_samples, latent_space_dim ):
    #### Dimensions of this model
    #### Input = 30 x 12 x 2
    #### Conv1 = 30 x 12 x 16
    #### Pool1 = 15 x 6 x 16 (2,2)
    #### Conv2 = 15 x 6 x 32
    #### Pool2 = 5 x 3 x 32 (3,2)
    #### Conv3 = 5 x 3 x 64
    #### Pool3 = 1 x 3 x 64 (5,1)
    #### Conv4 = 1 x 3 x 128
    #### Pool4 = 1 x 1 x 128 (1,3)
    #### Conv5 =batch x 1 x 1 x 64
    ###########################################################
    #### Height, width, and timesteps have limited flexibility
    #### within these parameters
    ###########################################################
    convKernel = (3,3) 
    poolKernel1 = (2,2) 
    poolKernel2 = (3,2) 
    poolKernel3 = (5,1) 
    poolKernel4 = (1,3) 
    input_1 = keras.layers.Input(( timesteps, width, height))
    encoder_1 = keras.layers.Conv2D(convFilters, convKernel, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='conv2d_1')(input_1)
    encoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_1')(encoder_1)
    encoder_1 = keras.layers.Activation('relu')(encoder_1)
    mp_out_1 = keras.layers.AveragePooling2D(pool_size=poolKernel1, data_format='channels_last', padding='same')(encoder_1)
    encoder_2 = keras.layers.Conv2D(convFilters*2, convKernel, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='conv2d_2')(mp_out_1)
    encoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_2')(encoder_2)
    encoder_2 = keras.layers.Activation('relu')(encoder_2)
    mp_out_2 = keras.layers.AveragePooling2D(pool_size=poolKernel2, data_format='channels_last', padding='same')(encoder_2)
    encoder_3 = keras.layers.Conv2D(convFilters*4, convKernel, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='conv2d_3')(mp_out_2)
    encoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_3')(encoder_3)
    encoder_3 = keras.layers.Activation('relu')(encoder_3)
    mp_out_3 = keras.layers.AveragePooling2D(pool_size=poolKernel3, data_format='channels_last', padding='same')(encoder_3)
    encoder_4 = keras.layers.Conv2D(convFilters*8, convKernel, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='conv2d_4')(mp_out_3)
    encoder_4 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_4')(encoder_4)
    encoder_4 = keras.layers.Activation('relu')(encoder_4)
    mp_out_4 = keras.layers.AveragePooling2D(pool_size=poolKernel4, data_format='channels_last', padding='same')(encoder_4)
    encoder_5 = keras.layers.Conv2D(convFilters*4, convKernel, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='conv2d_5')(mp_out_4)
    encoder_5 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_5')(encoder_5)
    encoder_5 = keras.layers.Activation('relu')(encoder_5)
    ################
    cluster_layer = custom_layers.ClusteringLayer( num_clusters, initial_clusters, num_samples, latent_space_dim, name='ClusterLayerOut')(encoder_5)
    ################
    decoder_1 = keras.layers.Conv2D(convFilters*4, convKernel,  data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='seg_conv2dTrans_1')(encoder_5)
    decoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_6')(decoder_1)
    decoder_1 = keras.layers.Activation('relu')(decoder_1)
    decoder_2 = keras.layers.Conv2DTranspose(convFilters*8, convKernel, strides=poolKernel4, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='seg_conv2dTrans_2')(decoder_1)
    decoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_7')(decoder_2)
    decoder_2 = keras.layers.Activation('relu')(decoder_2)
    decoder_3 = keras.layers.Conv2DTranspose(convFilters*4, convKernel, strides=poolKernel3, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='seg_conv2dTrans_3')(decoder_2)
    decoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_8')(decoder_3)
    decoder_3 = keras.layers.Activation('relu')(decoder_3)
    decoder_4 = keras.layers.Conv2DTranspose(convFilters*2, convKernel, strides=poolKernel2, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='seg_conv2dTrans_4')(decoder_3)
    decoder_4 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_9')(decoder_4)
    decoder_4 = keras.layers.Activation('relu')(decoder_4)
    decoder_5 = keras.layers.Conv2DTranspose(convFilters, convKernel, strides=poolKernel1, data_format='channels_last', padding='same', use_bias=False, kernel_initializer='he_normal',name='seg_conv2dTrans_5')(decoder_4)
    decoder_5 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_10')(decoder_5)
    decoder_5 = keras.layers.Activation('relu')(decoder_5)
    decoder_out = keras.layers.Conv2D(height, convKernel, data_format='channels_last', padding='same', use_bias=True, kernel_initializer='he_normal', name='conv2d_out')(decoder_5)
    return ( input_1, encoder_5 , cluster_layer, decoder_out )
'''
def encoder_decoder_singleFrame( height, width, convFilters, num_clusters, initial_clusters, num_samples, latent_space_dim ):
    #### Suggested dimensions of network for the syntatic dataset
    #### ConvFilters = 1
    #### Input =  12 x 2
    #### Conv1 = 12 x 1
    #### Pool1 = 6 x 1 (2)
    #### Conv2 = 6 x 2
    #### Pool2 = 3 x 2 (2)
    #### Conv3 = 3 x 4
    #### Pool3 = 1 x 4 (3)
    ############################
    #### ClusterLayer
    ############################
    #### Dec1 = 3 x 4
    #### Dec2 = 6 x 2
    #### Dec3 = 12 x 1
    #### DecOut = 12 x 2
    convKernel = (3) 
    poolKernel1 = (2) 
    poolKernel2 = (3) 
    input_1 = keras.layers.Input(( width, height))
    encoder_1 = keras.layers.Conv1D(convFilters, convKernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='conv2d_1')(input_1)
    encoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_1')(encoder_1)
    encoder_1 = keras.layers.Activation('relu')(encoder_1)
    mp_out_1 = keras.layers.AveragePooling1D(pool_size=poolKernel1, padding='same')(encoder_1)
    encoder_2 = keras.layers.Conv1D(convFilters*2, convKernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='conv2d_2')(mp_out_1)
    encoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_2')(encoder_2)
    encoder_2 = keras.layers.Activation('relu')(encoder_2)
    mp_out_2 = keras.layers.AveragePooling1D(pool_size=poolKernel1, padding='same')(encoder_2)
    encoder_3 = keras.layers.Conv1D(convFilters*4, convKernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='conv2d_3')(mp_out_2)
    encoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_3')(encoder_3)
    encoder_3 = keras.layers.Activation('relu')(encoder_3)
    mp_out_3 = keras.layers.AveragePooling1D(pool_size=poolKernel2, padding='same')(encoder_3)
    ################
    cluster_layer = custom_layers.ClusteringLayer( num_clusters, initial_clusters, num_samples, latent_space_dim, name='ClusterLayerOut')(mp_out_3)
    ################
    decoder_1 = keras.layers.Conv1D(convFilters*4, convKernel,padding='same', use_bias=False, kernel_initializer='he_normal',name='seg_conv2dTrans_2')(mp_out_3)
    decoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_7')(decoder_1)
    decoder_1 = keras.layers.Activation('relu')(decoder_1)
    decoder_1 = keras.layers.UpSampling1D(poolKernel2)(decoder_1)
    decoder_2 = keras.layers.Conv1D(convFilters*2, convKernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='seg_conv2dTrans_3')(decoder_1)
    decoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_8')(decoder_2)
    decoder_2 = keras.layers.Activation('relu')(decoder_2)
    decoder_2 = keras.layers.UpSampling1D(poolKernel1)(decoder_2)
    decoder_3 = keras.layers.Conv1D(convFilters, convKernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='seg_conv2dTrans_4')(decoder_2)
    decoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_9')(decoder_3)
    decoder_3 = keras.layers.Activation('relu')(decoder_3)
    decoder_3 = keras.layers.UpSampling1D(poolKernel1)(decoder_3)
    decoder_out = keras.layers.Conv1D(height, convKernel, padding='same', use_bias=True, kernel_initializer='he_normal', name='conv2d_out')(decoder_3)
    return ( input_1, mp_out_3, cluster_layer, decoder_out )
'''


def encoder_decoder_singleFrame( height, width, convFilters, num_clusters, initial_clusters, num_samples, latent_space_dim ):
    input_1 = keras.layers.Input(( width, height))
    flatten_1 = keras.layers.Flatten()( input_1 )
    encoder_1 = keras.layers.Dense(convFilters*4, use_bias=False, kernel_initializer='he_normal',name='dense_1')(flatten_1)
    encoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_1')(encoder_1)
    encoder_1 = keras.layers.Activation('relu')(encoder_1)
    encoder_2 = keras.layers.Dense(convFilters*3, use_bias=False, kernel_initializer='he_normal',name='dense_2')(encoder_1)
    encoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_2')(encoder_2)
    encoder_2 = keras.layers.Activation('relu')(encoder_2)
    encoder_3 = keras.layers.Dense(convFilters*2, use_bias=False, kernel_initializer='he_normal',name='dense_3')(encoder_2)
    encoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_3')(encoder_3)
    encoder_3 = keras.layers.Activation('relu')(encoder_3)
    encoder_4 = keras.layers.Dense(convFilters, use_bias=False, kernel_initializer='he_normal',name='dense_4')(encoder_3)
    encoder_4 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_4')(encoder_4)
    encoder_4 = keras.layers.Activation('relu')(encoder_4)
    ################
    cluster_layer = custom_layers.ClusteringLayer( num_clusters, initial_clusters, num_samples, latent_space_dim, name='ClusterLayerOut')(encoder_4)
    ################
    decoder_1 = keras.layers.Dense(convFilters*4,use_bias=False, kernel_initializer='he_normal',name='dense_5')(encoder_4)
    decoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_5')(decoder_1)
    decoder_1 = keras.layers.Activation('relu')(decoder_1)
    decoder_2 = keras.layers.Dense(convFilters*2, use_bias=False, kernel_initializer='he_normal',name='dense_6')(decoder_1)
    decoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_6')(decoder_2)
    decoder_2 = keras.layers.Activation('relu')(decoder_2)
    decoder_3 = keras.layers.Dense(convFilters, use_bias=False, kernel_initializer='he_normal',name='dense_7')(decoder_2)
    decoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_7')(decoder_3)
    decoder_3 = keras.layers.Activation('relu')(decoder_3)
    decoder_out = keras.layers.Dense( width* height, use_bias=True, kernel_initializer='he_normal', name='dense_out')(decoder_3)
    decoder_out = keras.layers.Reshape(( width, height ))(decoder_out)
    return ( input_1, encoder_4, cluster_layer, decoder_out )


def singleFrame_TimeDistributed( timesteps, height, width, convFilters, num_clusters, initial_clusters, num_samples, latent_space_dim ):
    from keras import backend as K
    K.set_learning_phase(1)
    input_1 = keras.layers.Input((timesteps, width, height))
    #####
    input_2 = keras.layers.Input(( width, height ))
    flatten_1 = keras.layers.Flatten()( input_2 )
    encoder_1 = keras.layers.Dense(convFilters*4, use_bias=False, kernel_initializer='he_normal',name='dense_1')(flatten_1)
    encoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_1')(encoder_1)
    encoder_1 = keras.layers.Activation('relu')(encoder_1)
    encoder_2 = keras.layers.Dense(convFilters*3, use_bias=False, kernel_initializer='he_normal',name='dense_2')(encoder_1)
    encoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_2')(encoder_2)
    encoder_2 = keras.layers.Activation('relu')(encoder_2)
    encoder_3 = keras.layers.Dense(convFilters*2, use_bias=False, kernel_initializer='he_normal',name='dense_3')(encoder_2)
    encoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_3')(encoder_3)
    encoder_3 = keras.layers.Activation('relu')(encoder_3)
    encoder_4 = keras.layers.Dense(convFilters, use_bias=False, kernel_initializer='he_normal',name='dense_4')(encoder_3)
    encoder_4 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_4')(encoder_4)
    encoder_4 = keras.layers.Activation('relu')(encoder_4)
    ################
    ################
    decoder_1 = keras.layers.Dense(convFilters*4,use_bias=False, kernel_initializer='he_normal',name='dense_5')(encoder_4)
    decoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_5')(decoder_1)
    decoder_1 = keras.layers.Activation('relu')(decoder_1)
    decoder_2 = keras.layers.Dense(convFilters*2, use_bias=False, kernel_initializer='he_normal',name='dense_6')(decoder_1)
    decoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_6')(decoder_2)
    decoder_2 = keras.layers.Activation('relu')(decoder_2)
    decoder_3 = keras.layers.Dense(convFilters, use_bias=False, kernel_initializer='he_normal',name='dense_7')(decoder_2)
    decoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_7')(decoder_3)
    decoder_3 = keras.layers.Activation('relu')(decoder_3)
    decoder_out = keras.layers.Dense( width* height, use_bias=True, kernel_initializer='he_normal', name='dense_out')(decoder_3)
    decoder_out = keras.layers.Reshape(( width, height ))(decoder_out)
    tD_model = keras.models.Model( 
                inputs = [ input_2 ],
                outputs = [  encoder_4, decoder_out ]
                )
    outs = []
    for out in tD_model.output:
        outs.append(keras.layers.TimeDistributed(keras.models.Model(tD_model.input,out))(input_1))

    encoder_Fout, decoder_Fout = outs
    cluster_layer = custom_layers.ClusteringLayer( num_clusters, initial_clusters, num_samples, latent_space_dim, name='ClusterLayerOut')(encoder_Fout)    
    return ( input_1, encoder_Fout, cluster_layer, decoder_Fout )

def singleFrame_Deep_TimeDistributed( timesteps, height, width, convFilters, num_clusters, initial_clusters, num_samples, latent_space_dim ):
    from keras import backend as K
    K.set_learning_phase(1)
    input_1 = keras.layers.Input((timesteps, width, height))
    #####
    input_2 = keras.layers.Input(( width, height ))
    flatten_1 = keras.layers.Flatten()( input_2 )
    encoder_1 = keras.layers.Dense(convFilters*10, use_bias=False, kernel_initializer='he_normal',name='dense_1')(flatten_1)
    encoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_1')(encoder_1)
    encoder_1 = keras.layers.Activation('relu')(encoder_1)
    encoder_2 = keras.layers.Dense(convFilters*9, use_bias=False, kernel_initializer='he_normal',name='dense_2')(keras.layers.concatenate([flatten_1,encoder_1]))
    encoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_2')(encoder_2)
    encoder_2 = keras.layers.Activation('relu')(encoder_2)
    encoder_3 = keras.layers.Dense(convFilters*8, use_bias=False, kernel_initializer='he_normal',name='dense_3')(keras.layers.concatenate([flatten_1,encoder_1,encoder_2]))
    encoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_3')(encoder_3)
    encoder_3 = keras.layers.Activation('relu')(encoder_3)
    encoder_4 = keras.layers.Dense(convFilters*7, use_bias=False, kernel_initializer='he_normal',name='dense_4')(keras.layers.concatenate([flatten_1,encoder_2,encoder_3]))
    encoder_4 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_4')(encoder_4)
    encoder_4 = keras.layers.Activation('relu')(encoder_4)
    encoder_5 = keras.layers.Dense(convFilters*6, use_bias=False, kernel_initializer='he_normal',name='dense_5')(keras.layers.concatenate([flatten_1,encoder_3,encoder_4]))
    encoder_5 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_5')(encoder_5)
    encoder_5 = keras.layers.Activation('relu')(encoder_5)
    encoder_6 = keras.layers.Dense(convFilters*5, use_bias=False, kernel_initializer='he_normal',name='dense_6')(keras.layers.concatenate([flatten_1,encoder_4,encoder_5]))
    encoder_6 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_6')(encoder_6)
    encoder_6 = keras.layers.Activation('relu')(encoder_6)
    encoder_7 = keras.layers.Dense(convFilters*4, use_bias=False, kernel_initializer='he_normal',name='dense_7')(keras.layers.concatenate([flatten_1,encoder_5,encoder_6]))
    encoder_7 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_7')(encoder_7)
    encoder_7 = keras.layers.Activation('relu')(encoder_7)
    encoder_8 = keras.layers.Dense(convFilters*3, use_bias=False, kernel_initializer='he_normal',name='dense_8')(keras.layers.concatenate([flatten_1,encoder_6,encoder_7]))
    encoder_8 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_8')(encoder_8)
    encoder_8 = keras.layers.Activation('relu')(encoder_8)
    encoder_9 = keras.layers.Dense(convFilters*2, use_bias=False, kernel_initializer='he_normal',name='dense_9')(keras.layers.concatenate([flatten_1,encoder_7,encoder_8]))
    encoder_9 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_9')(encoder_9)
    encoder_9 = keras.layers.Activation('relu')(encoder_9)
    encoder_10 = keras.layers.Dense(convFilters, use_bias=False, kernel_initializer='he_normal',name='dense_10')(keras.layers.concatenate([flatten_1,encoder_8,encoder_9]))
    encoder_10 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_10')(encoder_10)
    encoder_10 = keras.layers.Activation('relu')(encoder_10)
    ################
    ################
    decoder_1 = keras.layers.Dense(convFilters*10,use_bias=False, kernel_initializer='he_normal',name='dense_11')(encoder_10)
    decoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_11')(decoder_1)
    decoder_1 = keras.layers.Activation('relu')(decoder_1)
    decoder_2 = keras.layers.Dense(convFilters*9, use_bias=False, kernel_initializer='he_normal',name='dense_12')(decoder_1)
    decoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_12')(decoder_2)
    decoder_2 = keras.layers.Activation('relu')(decoder_2)
    decoder_3 = keras.layers.Dense(convFilters*8, use_bias=False, kernel_initializer='he_normal',name='dense_13')(decoder_2)
    decoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_13')(decoder_3)
    decoder_3 = keras.layers.Activation('relu')(decoder_3)
    decoder_4 = keras.layers.Dense(convFilters*7,use_bias=False, kernel_initializer='he_normal',name='dense_14')(decoder_3)
    decoder_4 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_14')(decoder_4)
    decoder_4 = keras.layers.Activation('relu')(decoder_4)
    decoder_5 = keras.layers.Dense(convFilters*6, use_bias=False, kernel_initializer='he_normal',name='dense_15')(decoder_4)
    decoder_5 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_15')(decoder_5)
    decoder_5 = keras.layers.Activation('relu')(decoder_5)
    decoder_6 = keras.layers.Dense(convFilters*5, use_bias=False, kernel_initializer='he_normal',name='dense_16')(decoder_5)
    decoder_6 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_16')(decoder_6)
    decoder_6 = keras.layers.Activation('relu')(decoder_6)
    decoder_7 = keras.layers.Dense(convFilters*4,use_bias=False, kernel_initializer='he_normal',name='dense_17')(decoder_6)
    decoder_7 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_17')(decoder_7)
    decoder_7 = keras.layers.Activation('relu')(decoder_7)
    decoder_8 = keras.layers.Dense(convFilters*3, use_bias=False, kernel_initializer='he_normal',name='dense_18')(decoder_7)
    decoder_8 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_18')(decoder_8)
    decoder_8 = keras.layers.Activation('relu')(decoder_8)
    decoder_9 = keras.layers.Dense(convFilters*2, use_bias=False, kernel_initializer='he_normal',name='dense_19')(decoder_8)
    decoder_9 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_19')(decoder_9)
    decoder_9 = keras.layers.Activation('relu')(decoder_9)
    decoder_10 = keras.layers.Dense(convFilters, use_bias=False, kernel_initializer='he_normal',name='dense_20')(decoder_9)
    decoder_10 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_20')(decoder_10)
    decoder_10 = keras.layers.Activation('relu')(decoder_10)
    decoder_out = keras.layers.Dense( width* height, use_bias=True, kernel_initializer='he_normal', name='dense_out')(decoder_10)
    decoder_out = keras.layers.Reshape(( width, height ))(decoder_out)
    tD_model = keras.models.Model( 
                inputs = [ input_2 ],
                outputs = [  encoder_10, decoder_out ]
                )
    outs = []
    for out in tD_model.output:
        outs.append(keras.layers.TimeDistributed(keras.models.Model(tD_model.input,out))(input_1))
    encoder_Fout, decoder_Fout = outs
    cluster_layer = custom_layers.ClusteringLayer( num_clusters, initial_clusters, num_samples, latent_space_dim, name='ClusterLayerOut')(encoder_Fout)    
    return ( input_1, encoder_Fout, cluster_layer, decoder_Fout )




'''
def singleFrame_Deep_TimeDistributed( timesteps, height, width, convFilters, num_clusters, initial_clusters, num_samples, latent_space_dim ):
    from keras import backend as K
    K.set_learning_phase(1)
    input_1 = keras.layers.Input((timesteps, width, height))
    #####
    input_2 = keras.layers.Input(( width, height ))
    flatten_1 = keras.layers.Flatten()( input_2 )
    encoder_1 = keras.layers.Dense(convFilters*10, use_bias=False, kernel_initializer='he_normal',name='dense_1')(flatten_1)
    encoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_1')(encoder_1)
    encoder_1 = keras.layers.Activation('relu')(encoder_1)
    encoder_2 = keras.layers.Dense(convFilters*9, use_bias=False, kernel_initializer='he_normal',name='dense_2')(encoder_1)
    encoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_2')(encoder_2)
    encoder_2 = keras.layers.Activation('relu')(encoder_2)
    encoder_3 = keras.layers.Dense(convFilters*8, use_bias=False, kernel_initializer='he_normal',name='dense_3')(encoder_2)
    encoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_3')(encoder_3)
    encoder_3 = keras.layers.Activation('relu')(encoder_3)
    encoder_4 = keras.layers.Dense(convFilters*7, use_bias=False, kernel_initializer='he_normal',name='dense_4')(encoder_3)
    encoder_4 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_4')(encoder_4)
    encoder_4 = keras.layers.Activation('relu')(encoder_4)
    encoder_5 = keras.layers.Dense(convFilters*6, use_bias=False, kernel_initializer='he_normal',name='dense_5')(encoder_4)
    encoder_5 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_5')(encoder_5)
    encoder_5 = keras.layers.Activation('relu')(encoder_5)
    encoder_6 = keras.layers.Dense(convFilters*5, use_bias=False, kernel_initializer='he_normal',name='dense_6')(encoder_5)
    encoder_6 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_6')(encoder_6)
    encoder_6 = keras.layers.Activation('relu')(encoder_6)
    encoder_7 = keras.layers.Dense(convFilters*4, use_bias=False, kernel_initializer='he_normal',name='dense_7')(encoder_6)
    encoder_7 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_7')(encoder_7)
    encoder_7 = keras.layers.Activation('relu')(encoder_7)
    encoder_8 = keras.layers.Dense(convFilters*3, use_bias=False, kernel_initializer='he_normal',name='dense_8')(encoder_7)
    encoder_8 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_8')(encoder_8)
    encoder_8 = keras.layers.Activation('relu')(encoder_8)
    encoder_9 = keras.layers.Dense(convFilters*2, use_bias=False, kernel_initializer='he_normal',name='dense_9')(encoder_8)
    encoder_9 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_9')(encoder_9)
    encoder_9 = keras.layers.Activation('relu')(encoder_9)
    encoder_10 = keras.layers.Dense(convFilters, use_bias=False, kernel_initializer='he_normal',name='dense_10')(encoder_9)
    encoder_10 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_10')(encoder_10)
    encoder_10 = keras.layers.Activation('relu')(encoder_10)
    ################
    ################
    decoder_1 = keras.layers.Dense(convFilters*10,use_bias=False, kernel_initializer='he_normal',name='dense_11')(encoder_10)
    decoder_1 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_11')(decoder_1)
    decoder_1 = keras.layers.Activation('relu')(decoder_1)
    decoder_2 = keras.layers.Dense(convFilters*9, use_bias=False, kernel_initializer='he_normal',name='dense_12')(decoder_1)
    decoder_2 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_12')(decoder_2)
    decoder_2 = keras.layers.Activation('relu')(decoder_2)
    decoder_3 = keras.layers.Dense(convFilters*8, use_bias=False, kernel_initializer='he_normal',name='dense_13')(decoder_2)
    decoder_3 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_13')(decoder_3)
    decoder_3 = keras.layers.Activation('relu')(decoder_3)
    decoder_4 = keras.layers.Dense(convFilters*7,use_bias=False, kernel_initializer='he_normal',name='dense_14')(decoder_3)
    decoder_4 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_14')(decoder_4)
    decoder_4 = keras.layers.Activation('relu')(decoder_4)
    decoder_5 = keras.layers.Dense(convFilters*6, use_bias=False, kernel_initializer='he_normal',name='dense_15')(decoder_4)
    decoder_5 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_15')(decoder_5)
    decoder_5 = keras.layers.Activation('relu')(decoder_5)
    decoder_6 = keras.layers.Dense(convFilters*5, use_bias=False, kernel_initializer='he_normal',name='dense_16')(decoder_5)
    decoder_6 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_16')(decoder_6)
    decoder_6 = keras.layers.Activation('relu')(decoder_6)
    decoder_7 = keras.layers.Dense(convFilters*4,use_bias=False, kernel_initializer='he_normal',name='dense_17')(decoder_6)
    decoder_7 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_17')(decoder_7)
    decoder_7 = keras.layers.Activation('relu')(decoder_7)
    decoder_8 = keras.layers.Dense(convFilters*3, use_bias=False, kernel_initializer='he_normal',name='dense_18')(decoder_7)
    decoder_8 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_18')(decoder_8)
    decoder_8 = keras.layers.Activation('relu')(decoder_8)
    decoder_9 = keras.layers.Dense(convFilters*2, use_bias=False, kernel_initializer='he_normal',name='dense_19')(decoder_8)
    decoder_9 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_19')(decoder_9)
    decoder_9 = keras.layers.Activation('relu')(decoder_9)
    decoder_10 = keras.layers.Dense(convFilters, use_bias=False, kernel_initializer='he_normal',name='dense_20')(decoder_9)
    decoder_10 = keras.layers.BatchNormalization(axis=-1, name='seg_batchNorm_20')(decoder_10)
    decoder_10 = keras.layers.Activation('relu')(decoder_10)
    decoder_out = keras.layers.Dense( width* height, use_bias=True, kernel_initializer='he_normal', name='dense_out')(decoder_10)
    decoder_out = keras.layers.Reshape(( width, height ))(decoder_out)
    tD_model = keras.models.Model( 
                inputs = [ input_2 ],
                outputs = [  encoder_10, decoder_out ]
                )
    outs = []
    for out in tD_model.output:
        outs.append(keras.layers.TimeDistributed(keras.models.Model(tD_model.input,out))(input_1))
    encoder_Fout, decoder_Fout = outs
    cluster_layer = custom_layers.ClusteringLayer( num_clusters, initial_clusters, num_samples, latent_space_dim, name='ClusterLayerOut')(encoder_Fout)    
    return ( input_1, encoder_Fout, cluster_layer, decoder_Fout )
'''
