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
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import euclidean
import imageio
import misc_functions
import cluster_models

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


###############################
###  Basic training loop
###############################
def default_training( parameters, Net_, trainLoad, validLoad ):
    Net_.summary()
    metric_list_size = len( Net_.metrics_names )
    cluster_out_label = np.zeros( ( int( parameters['batch_size'] ), 1 ), dtype = np.float32 )
    for i in range( parameters['epoch_num'] ):
        loss = np.zeros( ( 25, metric_list_size ) )
        for j in range( int( parameters['training_size'] ) ):
            seekBatch = True
            while seekBatch:
                try:
                    x_,y_ = trainLoad.q.popleft()
                    seekBatch = False
                except:
                    print('Queue empty')
                    sys.stdout.flush()
                    time.sleep(1)
    ######
            loss[j % 25,:] = Net_.train_on_batch( x_, [ cluster_out_label, x_ ] )#### Note that the first label input is necessary for keras, but is not used in the loss
            if( j is not 0 and j % 25 == 0 ):
                print( 'Loss: {0!s}'.format( np.average( loss, axis=0 ) ) )
                loss = np.zeros( ( 25, metric_list_size ) )
                sys.stdout.flush()
        for j in range( int( parameters['validation_size'] ) ):
            seekValBatch = True
            while seekValBatch:
                try:
                    xv_,yv_ = validLoad.q.popleft()
                    seekValBatch = False
                except:
                    print('Validation Queue empty')
                    sys.stdout.flush()
                    time.sleep(1)
                valLoss = Net_.test_on_batch( xv_, [ cluster_out_label, xv_ ] )#### Note that the first label input is necessary for keras, but is not used in the loss
                print( 'Validation: {0!s}'.format( valLoss ) )
        if( i % 10 == 0 or i == parameters['epoch_num']-1 ):
            savePath = '/home/gardij/Code/NN_models/PointClustering_epoch_'+str(i)+'_'+str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))+'.h5'
            Net_.save_weights(savePath)
            print( "Saved Model: {0!s}".format( savePath ) )
    return savePath

##################################################
###  Training Loop with cluster centroid updates
##################################################
def cluster_training( parameters, Net_, trainLoad, validLoad, clusterLayer ):
    metric_list_size = len( Net_.metrics_names )
    cluster_out_label = np.zeros( ( int( parameters['batch_size'] ), 1 ), dtype = np.float32 )
    latent_out_label = np.zeros(( int( parameters['batch_size'] ), int( parameters['timesteps']), int(parameters['convFilters']) ), dtype = np.float32 )
    for i in range( parameters['epoch_num'] ):
        loss = np.zeros( ( 25, metric_list_size ) )
        for j in range( int( parameters['training_size'] ) ):
            seekBatch = True
            while seekBatch:
                try:
                    x_,y_ = trainLoad.q.popleft()
                    seekBatch = False
                except:
                    print('Queue empty')
                    sys.stdout.flush()
                    time.sleep(1)
    ######
            loss[j % 25,:] = Net_.train_on_batch( x_, [latent_out_label, cluster_out_label, x_ ] )#### Note that the first and second label input is necessary for keras, but is not used in the loss
            if( j is not 0 and j % 25 == 0 ):
                print( 'Loss: {0!s}'.format( np.average( loss, axis=0 ) ) )
                loss = np.zeros( ( 25, metric_list_size ) )
                latent_out_label, clusterTemp, decodeTemp = Net_.predict_on_batch( x_ )
                sys.stdout.flush()
#            if( j is not 0 and j % 200 == 0):
#               kmeans_func = KMeans( n_clusters = parameters[ 'num_clusters' ], random_state = 0)
#                kmeans_out = kmeans_func.fit( np.reshape( latent_out_label, ( parameters['batch_size'], -1 ) ))
#                kmeans_centroids = kmeans_out.cluster_centers_
#                clusterLayer.updateClusterCenters( kmeans_centroids )              
        for j in range( int( parameters['validation_size'] ) ):
            seekValBatch = True
            while seekValBatch:
                try:
                    xv_,yv_ = validLoad.q.popleft()
                    seekValBatch = False
                except:
                    print('Validation Queue empty')
                    sys.stdout.flush()
                    time.sleep(1)
                valLoss = Net_.test_on_batch( xv_, [ latent_out_label, cluster_out_label, xv_ ] )#### Note that the first label input is necessary for keras, but is not used in the loss
                print( 'Validation: {0!s}'.format( valLoss ) )
        if( i % 10 == 0 or i == parameters['epoch_num']-1 ):
            savePath = '/home/gardij/Code/NN_models/PointClustering_epoch_'+str(i)+'_'+str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))+'.h5'
            Net_.save_weights(savePath)
            print( "Saved Model: {0!s}".format( savePath ) )
    return savePath



###############################
###  Train Model routines
###############################
def model_train_initial(parameters,trainLoad,validLoad):
    parameters['training_size'] = int( trainLoad.num_samples )
    parameters['validation_size'] = int( validLoad.num_samples )
    print('Total training samples: {0!s}\nTotal validation samples: {1!s}'.format(parameters['training_size'],parameters['validation_size'] ))
    parameters['training_size'] = int( parameters['training_size'] / parameters['batch_size'] )
    parameters['validation_size'] = int( parameters['validation_size'] / parameters['batch_size'] )
    if(parameters['model_type'] == 'timeChunk'):
        input_, latent_, cluster_, decoder_ = cluster_models.encoder_decoder_timechunk(
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['timesteps'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    elif(parameters['model_type'] == 'singleFrameTD'):
        input_, latent_, cluster_, decoder_ = cluster_models.singleFrame_TimeDistributed(
                                                    parameters['timesteps'],
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    elif(parameters['model_type'] == 'singleFrameTDDeep'):
        input_, latent_, cluster_, decoder_ = cluster_models.singleFrame_Deep_TimeDistributed(
                                                    parameters['timesteps'],
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    else:
        print('Model type argument is required. Did you define your own model? If so, modify network_routines.py. Current Model type is: {0!s}'.format(parameters['model_type']))
        quit()
    decoderLossWeight = K.variable(1.)
    clusterLossWeight = K.variable(0.)
    Net_ = keras.models.Model(
                inputs = [ input_ ],
                outputs = [ cluster_, decoder_ ]
                )
    Net_.compile(
                    optimizer = keras.optimizers.Adam(
                                                    lr = parameters['learning_rate']
                                                    ),
                    loss = [ misc_functions.cluster_hardening_loss, 'mse' ],
                    loss_weights = [ clusterLossWeight, decoderLossWeight ]
                )
    path = default_training( parameters, Net_, trainLoad, validLoad )
    return path

def model_train_withInitialization(parameters, modelpath, trainLoad, validLoad):
    ##################################################
    ### Import pretrained model
    ##################################################
    parameters['training_size'] = int( trainLoad.num_samples )
    parameters['validation_size'] = int( validLoad.num_samples )
    print('Total training samples: {0!s}\nTotal validation samples: {1!s}'.format(parameters['training_size'],parameters['validation_size'] ))
    parameters['training_size'] = int( parameters['training_size'] / parameters['batch_size'] )
    parameters['validation_size'] = int( parameters['validation_size'] / parameters['batch_size'] )
    if(parameters['model_type'] == 'timeChunk'):
        input_, latent_, cluster_, decoder_ = cluster_models.encoder_decoder_timechunk(
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['timesteps'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    elif(parameters['model_type'] == 'singleFrameTD'):
        input_, latent_, cluster_, decoder_ = cluster_models.singleFrame_TimeDistributed(
                                                    parameters['timesteps'],
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    elif(parameters['model_type'] == 'singleFrameTDDeep'):
        input_, latent_, cluster_, decoder_ = cluster_models.singleFrame_Deep_TimeDistributed(
                                                    parameters['timesteps'],
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    else:
        print('Model type argument is required. Did you define your own model? If so, modify network_routines.py. Current Model type is: {0!s}'.format(parameters['model_type']))
        quit()
    Net_ = keras.models.Model(
                inputs = [ input_ ],
                outputs = [ latent_, cluster_, decoder_ ]
                )
    try:
        Net_.load_weights( modelpath, by_name=True)
        print('Loaded model: {0!s}'.format(modelpath))
    except:
        print('Could not load the passed modelpath into your current network choice. {0!s}'.format( modelpath))
        quit()
    ####################################################################################################
    ######## Use initial model of network to find the current kmeans cluster centers
    ######## Pass these cluster centers into the clustering layer
    ######## This information will be used to begin effective cluster assignment hardening loss updates
    ####################################################################################################
    print('Performing inferences on data with current model')
    sys.stdout.flush()
    initial_latent_space = []
    for element in grouper(trainLoad.data[:][:],parameters['batch_size']):
        try:
            curr_batch = np.asarray(list(zip(*element))[0])  ## The last batch doesn't work TODO Fix this (fillvalue?)
        except Exception:
            pass
        if(curr_batch.shape[0] == parameters['batch_size']):
            latentVectors,clusterLossOut,decoderOut = Net_.predict(curr_batch,batch_size=curr_batch.shape[0])
            initial_latent_space.append( np.reshape(latentVectors,(parameters['batch_size'],-1)) )
    ###This currently works, but is slow
#	for element in trainLoad.data:
#		latentVector, clusterLossOut, decoderOut  = Net_.predict( np.expand_dims(element[0],axis=0), batch_size = 1)
#		initial_latent_space.append( np.squeeze(latentVector) )
    print('Starting Kmeans.')
    current_latent_space = np.vstack(initial_latent_space)
    kmeans_func = KMeans( n_clusters = parameters[ 'num_clusters' ], random_state = 0)
    kmeans_out = kmeans_func.fit( current_latent_space )
    kmeans_centroids = kmeans_out.cluster_centers_
    latentLossWeight = K.variable(0.)
    decoderLossWeight = K.variable(1.)
    clusterLossWeight = K.variable(1.)
    Net_.compile(
                    optimizer = keras.optimizers.Adam(
                                                    lr = parameters[ 'learning_rate' ]
                                                    ),
                    loss = ['mse', misc_functions.cluster_hardening_loss, 'mse' ],
                    loss_weights = [latentLossWeight, clusterLossWeight, decoderLossWeight ]
                )
    try:
        Net_.load_weights( modelpath ,by_name=True)
        print('Loaded model: {0!s}'.format(modelpath))
    except:
        print('Could not load the passed modelpath into your current network choice. {0!s}'.format( modelpath))
        quit()
    print('Finished. Updating cluster centers.')
    curr_ClusterLayer = Net_.get_layer('ClusterLayerOut')
    curr_ClusterLayer.updateClusterCenters( kmeans_centroids )
    Net_.summary()
    print('Finished. Starting Training.')
    path = cluster_training( parameters, Net_, trainLoad, validLoad, curr_ClusterLayer )
    return path


def model_infer_general( parameters, modelpath, outputpathprefix, trainLoad, validLoad ):
    ##################################################
    ### Import pretrained model
    ### '/home/gardij/Code/NN_models/PointClustering_CAH_epoch_1425_18-02-18-10-09-42.h5'
    ##################################################
    parameters['training_size'] = int( trainLoad.num_samples )
    parameters['validation_size'] = int( validLoad.num_samples )
    print('Total training samples: {0!s}\nTotal validation samples: {1!s}'.format(parameters['training_size'],parameters['validation_size'] ))
    parameters['training_size'] = int( parameters['training_size'] / parameters['batch_size'] )
    parameters['validation_size'] = int( parameters['validation_size'] / parameters['batch_size'] )
    ######################
    if( parameters['model_type'] == 'timeChunk' ):
        input_, latent_, cluster_, decoder_ = cluster_models.encoder_decoder_timechunk(
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['timesteps'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    elif(parameters['model_type'] == 'singleFrameTD'):
        input_, latent_, cluster_, decoder_ = cluster_models.singleFrame_TimeDistributed(
                                                    parameters['timesteps'],
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    elif(parameters['model_type'] == 'singleFrameTDDeep'):
        input_, latent_, cluster_, decoder_ = cluster_models.singleFrame_Deep_TimeDistributed(
                                                    parameters['timesteps'],
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    else:
        print('Model type argument is required. Did you define your own model? If so, modify network_routines.py. Current Model type is: {0!s}'.format(parameters['model_type']))
        quit()
    Net_ = keras.models.Model(
                inputs = [ input_ ],
                outputs = [ latent_ ]
                )
    try:
        Net_.load_weights( modelpath ,by_name=True)
        print('Loaded model: {0!s}'.format(modelpath))
    except:
        print('Could not load the passed modelpath into your current network choice. {0!s}'.format( modelpath))
        quit()
    ####################################################################################################
    ######## Use initial model of network to find the current kmeans cluster centers
    ######## Pass these cluster centers into the clustering layer
    ######## This information will be used to begin effective cluster assignment hardening loss updates
    ####################################################################################################
    initial_latent_space = []
    for element in grouper(trainLoad.data[:][:],parameters['batch_size']):
        try:
            curr_batch = np.asarray(list(zip(*element))[0])  
        except Exception:
            print('Error in loading data')
        latentVectors = Net_.predict(curr_batch,batch_size=curr_batch.shape[0])
        initial_latent_space.append( np.reshape(latentVectors,(curr_batch.shape[0],-1)) )
    print( 'Starting Kmeans.' )
    latent_space_out = np.concatenate(initial_latent_space,axis=0)
    np.save(str(outputpathprefix)+'_latent_space.npy', latent_space_out)
    kmeans_func = KMeans( n_clusters = parameters[ 'num_clusters' ], random_state = 0)
    kmeans_out = kmeans_func.fit( latent_space_out )  ### outputs sklearn kmeans object
    np.save(str(outputpathprefix)+'_kmeans_centroids.npy', kmeans_out.cluster_centers_) ### shape ==> (num_clusters, dimension of latent space)
    np.save(str(outputpathprefix)+'_kmeans_labels.npy', kmeans_out.labels_)
    print( 'Finished Kmeans. Starting plotting:')
############
    def find(lst, a):
        return [i for i, x in enumerate(lst) if x == a]
    for i in range(parameters['num_clusters']):
        fig = plt.figure(figsize=(20,12))
        misc_functions.plot_ClusterPosture(parameters,trainLoad,
                            find(kmeans_out.labels_,i),
                            fig,
                            str(outputpathprefix)+'all_syntactic_averagePosture_cluster'+str(i)+'.mp4'
                            )
        plt.close('all')
        print( 'Cluster plot {0!s} finished'.format(i))
    random_indices = np.random.randint(latent_space_out.shape[0], size=(np.min([2500,latent_space_out.shape[0]])))
    subsampled_data =  latent_space_out[random_indices]
    xLabels = ["Cluster {0!s}".format(x) for x in range(0,parameters['num_clusters'])]
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    misc_functions.tsne_clustering(subsampled_data,kmeans_out.labels_[random_indices],fig,ax,parameters['num_clusters'])
    fig.savefig(str(outputpathprefix)+'tSNE_kmeans_labels.png')
    print( 'Finished inference!' )


def output_cluster_video( parameters , predictData , modelpath, outputpath, labels = False):
    ######################
    if( parameters['model_type'] == 'timeChunk' ):
        input_, latent_, cluster_, decoder_ = cluster_models.encoder_decoder_timechunk(
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['timesteps'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    elif(parameters['model_type'] == 'singleFrameTD'):
        input_, latent_, cluster_, decoder_ = cluster_models.singleFrame_TimeDistributed(
                                                    parameters['timesteps'],
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    elif(parameters['model_type'] == 'singleFrameTDDeep'):
        input_, latent_, cluster_, decoder_ = cluster_models.singleFrame_Deep_TimeDistributed(
                                                    parameters['timesteps'],
                                                    parameters['height'],
                                                    parameters['width'],
                                                    parameters['convFilters'],
                                                    parameters['num_clusters'],
                                                    parameters['initial_clusters'],
                                                    parameters['num_samples'],
                                                    parameters['latent_space_dim']
                                                    )
    else:
        print('Model type argument is required. Did you define your own model? If so, modify network_routines.py. Current Model type is: {0!s}'.format(parameters['model_type']))
        quit()
    Net_ = keras.models.Model(
                inputs = [ input_ ],
                outputs = [ latent_ ]
                )
    try:
        Net_.load_weights( modelpath ,by_name=True)
        print('Loaded model: {0!s}'.format(modelpath))
    except:
        print('Could not load the passed modelpath into your current network choice. {0!s}'.format( modelpath))
        quit()
    #########################KMEANS########################
#    Net_.summary()
#    cluster_layer = Net_.get_layer('ClusterLayerOut')
#    clusterCenters, = cluster_layer.get_weights()
#    Net_ = keras.models.Model(
#                inputs = [ input_ ],
#                outputs = [ latent_ ]
#                )
    initial_latent_space = []
    initial_labels = []
    for element in grouper(predictData.data[:][:],parameters['batch_size']):
        try:
            curr_batch = np.asarray(list(zip(*element))[0])  
        except Exception:
            print('Error in loading data')
        latentVectors = Net_.predict(curr_batch,batch_size=curr_batch.shape[0])
        initial_latent_space.append( np.reshape(latentVectors,(curr_batch.shape[0],-1)) )
    print( 'Starting Kmeans.' )
    latent_space_out = np.concatenate(initial_latent_space,axis=0)
    kmeans_func = KMeans( n_clusters = parameters[ 'num_clusters' ], random_state = 0)
    kmeans_out = kmeans_func.fit( latent_space_out )  ### outputs sklearn kmeans object
    clusterCenters = kmeans_out.cluster_centers_
    try:
        Net_.load_weights( modelpath ,by_name=True)
        print('Loaded model: {0!s}'.format(modelpath))
    except:
        print('Could not load the passed modelpath into your current network choice. {0!s}'.format( modelpath))
        quit()
    writer = imageio.get_writer(outputpath+'_out.avi', fps=5)
    for element in predictData.data:
        latentVector = np.ndarray.flatten( Net_.predict( np.expand_dims(element[0],axis=0), batch_size = 1) )
        distance = []
        for i in range(clusterCenters.shape[0]):
            distance.append(euclidean(latentVector,clusterCenters[i,:]))
        val, idx = min((val, idx) for (idx, val) in enumerate(distance))
        misc_functions.inferClusterPlot( writer, idx, parameters['num_clusters'], element[1] )
        initial_labels.append(element[2])
    writer.close()
    random_indices = np.random.randint(latent_space_out.shape[0], size=(np.min([2500,latent_space_out.shape[0]])))
    subsampled_data =  latent_space_out[random_indices]
    print('shapes: {0!s}  {1!s}'.format(np.asarray(initial_labels).shape, latent_space_out.shape))
    subsampled_labels = np.asarray(initial_labels)[random_indices]
    xLabels = ["Cluster {0!s}".format(x) for x in range(0,parameters['num_clusters'])]
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    if(labels):
        labelNames_ = ['Not Grooming','Paw Licking','Face Washing','Flank Licking','Genital Grooming','Tail Grooming','Leg Grooming']
        misc_functions.tsne_clustering(subsampled_data,subsampled_labels,fig,ax,7,labelNames=labelNames_)
    else:
        misc_functions.tsne_clustering(subsampled_data,kmeans_out.labels_[random_indices],fig,ax,parameters['num_clusters'])
    fig.savefig(outputpath + '_tSNE_kmeans_labels.png')
