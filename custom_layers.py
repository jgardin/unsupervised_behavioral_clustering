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

import misc_functions

##########################################
###  Custom Layers
##########################################


class ClusteringLayer( keras.layers.Layer ):
    '''
    This layer gives soft assignments for the clusters based on distance from k-means based
    cluster centers. The weights of the layers are the cluster centers so that they can be learnt
    while optimizing for loss
    updateClusterCenters can be used to reassign the initial clustering centers (i.e. every epoch
    after the cluster assignment hardening loss has started to focus the clusters)
    '''
    def __init__( self, num_clusters, initial_clusters, num_samples, latent_space_dim, **kwargs ):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.num_clusters = num_clusters
        self.num_samples = num_samples
        self.latent_space_dim = latent_space_dim
        self.initial_clusters = initial_clusters
####
    def build( self, input_shape ):
        self.W = self.add_weight(shape = ( self.num_clusters, self.latent_space_dim ), name='ClusterCenters', initializer=keras.initializers.Zeros(), trainable=True )
        K.set_value(self.W,self.initial_clusters)
        super( ClusteringLayer, self ).build( self.latent_space_dim )
####
    def call( self, x ):
        return misc_functions.getSoftAssignments( x, self.W, self.num_clusters, self.latent_space_dim, self.num_samples )
####
    def updateClusterCenters( self, new_clusters ):
        K.set_value( self.W, new_clusters )
####
    def compute_output_shape( self, input_shape ):
        return ( self.num_samples, 1 )
