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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import misc_functions
import cluster_models
import custom_layers
import cluster_datasets
import network_routines


def main():
    #####################################
    ###  Argparse  Options
    ###   TODO improve this documentation
    ###  --dataset ==> [H5py file of dataset]
    ###  --initialize  ==> [ initializes weights of encoder decoder and trains on mean squared error loss only ]
    ###  --clusterHardening  ==> [ initializes weights of encoder decoder with a predefined model and trains on mean squared error loss and cluster hardening ]
    ###  --inferSyntactic
    #####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action = 'store_true', help = 'train initial encoder-decoder with basic weight initialization')
    parser.add_argument('--clusterHardening', action = 'store_true', help = 'train predefined encoder-decoder timechunk with MSE and Cluster Assignment Hardening')
    parser.add_argument('--startToFinishVideolist', action = 'store_true',  help = 'Run a inferred video list from start to finish. Pass in a path to a directory containing files. Mutually exclusive with --dataset.')
    parser.add_argument('--inferGeneral', action = 'store_true', help = 'Run a model on a dataset directory')
    parser.add_argument('--inferCluster', action = 'store_true', help = 'Run a model on a video and output predictions for all frames')
    ####################### NOTE ##############################################################################################################
    ###  To add or remove a model type, you add the new model in cluster_models.py
    ###  AND you change every function in network_routines.py to handle that new option.
    ###  YOU MIGHT need to a statement to the if/else chain in this file, depending on what you need to do.
    ###########################################################################################################################################
    parser.add_argument('--modelType', required = True, help = 'Type of model to train. Options are: timeChunk, singleFrameTD ,singleFrameTDDeep')
    parser.add_argument('--modelPath', help = 'Path to desired model weights h5 file')
    parser.add_argument('--labels', help = 'Path to labels file', default=False)
    ###########################################################################################################################################
    parser.add_argument('--dataset', help = 'path to dataset directory/file')
    parser.add_argument('--output',help = 'used with inference of a dataset, filepath to store transformed data')
    parser.add_argument('--numCluster', help = 'Number of clusters to predict on (hyperparameter)', default = 20)
    parser.add_argument('--learningRate', help = 'Learning rate of the training', default = 1e-4)
    parser.add_argument('--numEpochs', help = 'Max number of epochs to run', default = 1001)
    parser.add_argument('--numThreads', help = 'Maximum number of threads to use. Must be larger than one. Should be smaller than total number available on your system.', default = 2)
    parser.add_argument('--queueSize', help = 'Threading parameter. Larger number takes more memory but may improve speed. Suggested range: 10-1000', default = 200)
    parser.add_argument('--convFilters', help = 'Number of convolutional filters for the first layer of the desired model', default = 16)
    parser.add_argument('--latentSpaceDim', help = 'The total flattened dimension of the clustering layer input for the current model', default = 64)
    parser.add_argument('--timeSteps', help = 'The number of timesteps to analyze. Defaults to single frame', default = 1)
    parser.add_argument('--batchSize', help = 'Batch Size', default = 256)
    parser.add_argument('--augmentSize', help = 'augmentation Size', default = 0)
    parser.parse_args()
    args = parser.parse_args()
    ############ Turn on logging for processor threading debugging
    logging.basicConfig( level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
    ########################################################################
    ###  Model parameters
    ###  Dataset path
    ### '/home/gardij/Code/point_tracker_withSegmentation/segment_pointtracking_dataset/SYNgrooming_2017-12-11_infer_2018-2-8.h5'
    ########################################################################
    parameters = {
        'batch_size' : int(args.batchSize) ,
        'width' : 9 ,
        'height' : 2 ,
        'timesteps' : int(args.timeSteps) ,
        'num_labels' : 10 ,
        'learning_rate' : float(args.learningRate) ,
        'dataset' : str(args.dataset) ,
        'epoch_num' : int(args.numEpochs) ,
        'num_threads' : int(args.numThreads) , ### Must be greater than one
        'convFilters' : int(args.convFilters) ,
        'num_clusters' : int(args.numCluster) ,
        'latent_space_dim' : int(args.latentSpaceDim) ,
        'model_type' : str(args.modelType) ,
        'augment' : float(args.augmentSize),
        'model_path' : str(args.modelPath),
        'queueSize' : int(args.queueSize)
        }
    parameters['num_samples'] = parameters['batch_size']  ##This must be true for the model to function
    parameters['initial_clusters'] = np.random.random( ( parameters['num_clusters'], parameters['latent_space_dim'] ) ).astype(np.float32)
    if( parameters['num_threads'] < 2 ):
        print('Must use more than one thread')
        quit()
    #####################################
    ### Load datasets
    ### must be a h5 file with training and validation groups
    ### or a directory of video inferences
    #####################################
    if (os.path.isfile(args.dataset)):
        f = h5py.File(parameters['dataset'],libver='latest')
        h5py_handle_list = [f['training'], f['validation']]
    elif(os.path.isdir(args.dataset)):
        filelist = []
        h5py_handle_list = []
        for file in os.listdir(args.dataset):
            if file.endswith(".h5"):
                filelist.append(
                    os.path.join(args.dataset, file)
                )
        for filepath in filelist:
            h5py_handle_list.append( 
                h5py.File( filepath, 'r', libver='latest' ) 
            )
        if(len(h5py_handle_list) == 1): #### the user input a directory containing one file
            print('***Warning**** only one file was found in the given directory. Validation data is the same as training data in this run')
            h5py_handle_list[1] = h5py_handle_list[0] ## for the normal pipelines to work, there must be at least one validation example and one training example
    else:
        print('Error, the --dataset parameter is neither a file or directory: {0!s}'.format(args.dataset))
    if( args.inferCluster):
        videoData = cluster_datasets.clustering_video_Dataset(
                                        h5py_handle_list,
                                        parameters['batch_size'],
                                        parameters['height'],
                                        parameters['width'],
                                        parameters['timesteps'],
                                        augment = parameters['augment'],
                                        num_proc = parameters['num_threads']-1,
                                        queueSize = parameters['queueSize'],
                                        labels = args.labels
                                        )
        if(args.labels):
            network_routines.output_cluster_video( parameters, videoData , args.modelPath, args.output, args.labels )
        else:
            network_routines.output_cluster_video( parameters, videoData , args.modelPath, args.output )
        quit()

    trainingData = cluster_datasets.inference_video_Dataset(
                                        h5py_handle_list[:-1],
                                        parameters['batch_size'],
                                        parameters['height'],
                                        parameters['width'],
                                        parameters['timesteps'],
                                        augment = parameters['augment'],
                                        num_proc = parameters['num_threads']-1,
                                        queueSize = parameters['queueSize']
                                        )
    validationData = cluster_datasets.inference_video_Dataset(
                                        [h5py_handle_list[-1]],
                                        parameters['batch_size'],
                                        parameters['height'],
                                        parameters['width'],
                                        parameters['timesteps'],
                                        augment = 0,
                                        num_proc = parameters['num_threads']-1,
                                        queueSize = parameters['queueSize']
                                        )
    #####################################
    ####  Start desired functions
    #####################################
    if(args.initialize):
        network_routines.model_train_initial( parameters, trainingData, validationData)
    elif(args.clusterHardening and os.path.isfile(args.modelPath)):
        network_routines.model_train_withInitialization( parameters, args.modelPath, trainingData, validationData )
    elif(args.inferGeneral and os.path.isfile(args.modelPath)):
        network_routines.model_infer_general( parameters, str(args.modelPath), args.output, trainingData, validationData )
    elif( args.startToFinishVideolist  ):
        final_model = network_routines.model_train_initial( parameters, trainingData, validationData )
        final_model = network_routines.model_train_withInitialization( parameters, str(final_model), trainingData, validationData )
        network_routines.model_infer_general( parameters, str(final_model), args.output, trainingData, validationData )
    else:        
        print('Model type argument is required. Did you define your own model? If so, modify main.py. Current Model type is: {0!s}'.format(parameters['model_type']))
        quit()

                    

if __name__ == "__main__":
        main()

