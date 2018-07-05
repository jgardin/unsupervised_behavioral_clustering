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
import os
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

############################################################################################################################
###  Datasets:
###		These classes are responsible for loading the dataset and for starting the queue for the batches
############################################################################################################################

class syntactic_Dataset():
    ###########
    def __init__( self, h5py_handle, batch_size, height, width, timesteps, num_labels, augment=1, num_proc=1, queueSize=200 ):
        self.h5py_handle = h5py_handle
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.timesteps = timesteps
        self.num_labels = num_labels
        self.augment = augment
        self.videoIDs = [i for i in h5py_handle]
        self.data = []
        self.num_samples = 0
        self.num_proc = num_proc
        self.queueSize = queueSize
        self.q = deque([],queueSize)
        self.parse_dataset()
        self.start_workers()
    ############
    def parse_dataset( self ):
        for j in range(len(self.videoIDs)):
            for k in range(self.h5py_handle[self.videoIDs[j]+'/transformed_points'].shape[0]-self.timesteps):
                if(self.h5py_handle[self.videoIDs[j]+'/mask'][k+self.timesteps] == True):
                    self.data.append(
                        (
                            self.h5py_handle[self.videoIDs[j]+'/transformed_points'][k:k+self.timesteps,:,:].astype(np.float32),
                            keras.utils.to_categorical(np.argmax(np.bincount(self.h5py_handle[self.videoIDs[j]+'/labels'][k:k+self.timesteps])),num_classes=self.num_labels)
                        )
                    )
        self.num_samples = len(self.data)
    ############
    def data_loader( self ):
        data_copy = self.data.copy()
        x = np.zeros( ( self.batch_size, self.timesteps, self.width, self.height ), dtype=np.float32 )
        y = np.zeros( ( self.batch_size, self.num_labels ), dtype=np.float32 )
        while True:
            for i in range(self.batch_size):
                curr_index = np.random.randint(self.num_samples)
                x[i,...] = data_copy[curr_index][0]
                y[i,...] = data_copy[curr_index][1]
            if self.augment:
                x = np.add( x, np.random.normal( loc=0., scale=self.augment, size=x.shape ).astype( np.float32 ) )
            yield x, y
    def start_workers( self ):
        def worker(q,gen_func):
            x_ = np.zeros( ( self.batch_size, self.timesteps, self.width, self.height ), dtype=np.float32 )
            y_ = np.zeros( ( self.batch_size, self.num_labels ), dtype=np.float32 )
            while True:
                try:
                    x_, y_ = next( gen_func )
                except:
                    logging.exception('Problem in loading from generator')
                try:
                    self.q.appendleft( ( x_, y_ ) )
                except:
                    logging.exception('Problem in adding to the queue')
        threads = []
        loaders = []
        for i in range( self.num_proc ):
            loaders.append( self.data_loader() )
            threads.append( threading.Thread( target = worker, args= ( self.q, loaders[i] ) ) )
            threads[i].setDaemon( True )
            threads[i].start()


class syntactic_Dataset_SingleFrame():
    ###########
    def __init__( self, h5py_handle, batch_size, height, width, num_labels, augment=1, num_proc=1, queueSize=200 ):
        self.h5py_handle = h5py_handle
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_labels = num_labels
        self.augment = augment
        self.videoIDs = [i for i in h5py_handle]
        self.data = []
        self.num_samples = 0
        self.num_proc = num_proc
        self.queueSize = queueSize
        self.q = deque([],queueSize)
        self.parse_dataset()
        self.start_workers()
    ############
    def parse_dataset( self ):
        for j in range(len(self.videoIDs)):
            for k in range(self.h5py_handle[self.videoIDs[j]+'/transformed_points'].shape[0]):
                if(self.h5py_handle[self.videoIDs[j]+'/mask'][k] == True):
                    self.data.append(
                        (
                            self.h5py_handle[self.videoIDs[j]+'/transformed_points'][k,0:9,:].astype(np.float32),
                            keras.utils.to_categorical(self.h5py_handle[self.videoIDs[j]+'/labels'][k],num_classes=self.num_labels)
                        )
                    )
        self.num_samples = len(self.data)
    ############
    def data_loader( self ):
        data_copy = self.data.copy()
        x = np.zeros( ( self.batch_size, self.width, self.height ), dtype=np.float32 )
        y = np.zeros( ( self.batch_size, self.num_labels ), dtype=np.float32 )
        while True:
            for i in range(self.batch_size):
                curr_index = np.random.randint(self.num_samples)
                x[i,...] = data_copy[curr_index][0]
                y[i,...] = data_copy[curr_index][1]
            if self.augment:
                x = np.add( x, np.random.normal( loc=0., scale=self.augment, size=x.shape ).astype( np.float32 ) )
            yield x, y
    def start_workers( self ):
        def worker(q,gen_func):
            x_ = np.zeros( ( self.batch_size, self.width, self.height ), dtype=np.float32 )
            y_ = np.zeros( (1,), dtype=np.float32 )
            while True:
                try:
                    x_, y_ = next( gen_func )
                except:
                    logging.exception('Problem in loading from generator')
                try:
                    self.q.appendleft( ( x_, y_ ) )
                except:
                    logging.exception('Problem in adding to the queue')
        threads = []
        loaders = []
        for i in range( self.num_proc ):
            loaders.append( self.data_loader() )
            threads.append( threading.Thread( target = worker, args= ( self.q, loaders[i] ) ) )
            threads[i].setDaemon( True )
            threads[i].start()



class inference_video_Dataset():
    ###########
    def __init__( self, handle_list, batch_size, height, width, timesteps, augment=1, num_proc=1, queueSize=200):
        self.handle_list = handle_list
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.timesteps = timesteps
        self.augment = augment
        self.data = []
        self.num_samples = 0
        self.num_proc = num_proc
        self.queueSize = queueSize
        self.q = deque([],queueSize)
        self.parse_dataset()
        self.start_workers()
    ############
    def parse_dataset( self ):
        for i in range(len(self.handle_list)):
            h5py_handle = self.handle_list[i]
            videoIDs = [i for i in h5py_handle]
            for j in range(len(videoIDs)):
                for k in range(0,h5py_handle[videoIDs[j]+'/transformed_points'].shape[0]-self.timesteps,int(self.timesteps/2)):
                    if(np.sum(h5py_handle[videoIDs[j]+'/confidence'][k:k+self.timesteps,...]) > 50*(k+self.timesteps-k) ): ## Values of 100 or less seem to always mean there is no mouse in the frame
                        self.data.append(
                            (
                                h5py_handle[videoIDs[j]+'/transformed_points'][k:k+self.timesteps,0:9,:].astype(np.float32),
                            )
                        )
        self.num_samples = len(self.data)
    ############
    def data_loader( self ):
        data_copy = self.data.copy()
        x =  np.zeros( ( self.batch_size, self.timesteps, self.width, self.height ), dtype=np.float32 )
        while True:
            for i in range(self.batch_size):
                curr_index = np.random.randint(self.num_samples)
                x[i,...] = data_copy[curr_index][0]
            if self.augment:
                x = np.add( x, np.random.normal( loc=0., scale=self.augment, size=x.shape ).astype( np.float32 ) )
            yield x, 1.
    def start_workers( self ):
        def worker(q,gen_func):
            x_ = np.zeros( ( self.batch_size, self.timesteps, self.width, self.height ), dtype=np.float32 )
            y_ = np.zeros( 1, dtype=np.float32 )
            while True:
                try:
                    x_, y_ = next( gen_func )
                except:
                    logging.exception('Problem in loading from generator')
                try:
                    self.q.appendleft( ( x_, y_ ) )
                except:
                    logging.exception('Problem in adding to the queue')
        threads = []
        loaders = []
        for i in range( self.num_proc ):
            loaders.append( self.data_loader() )
            threads.append( threading.Thread( target = worker, args= ( self.q, loaders[i] ) ) )
            threads[i].setDaemon( True )
            threads[i].start()


class clustering_video_Dataset:
    ###########
    def __init__( self, handle_list, batch_size, height, width, timesteps, augment=1, num_proc=1, queueSize=200, labels=False):
        self.handle_list = handle_list
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.timesteps = timesteps
        self.augment = augment
        self.data = []
        self.num_samples = 0
        self.num_proc = num_proc
        self.queueSize = queueSize
        self.q = deque([],queueSize)
        if os.path.isfile(labels):
            self.labels = np.loadtxt(labels,delimiter='\t',dtype=np.float32)
            self.labelflag = True
        else:
            self.labelflag = False
        self.parse_dataset()

    ############
    def parse_dataset( self ):
        if(self.labelflag):
            for i in range(len(self.handle_list)):
                h5py_handle = self.handle_list[i]
                videoIDs = [i for i in h5py_handle]
                for j in range(len(videoIDs)):
                    for k in range(0,h5py_handle[videoIDs[j]+'/transformed_points'].shape[0]-self.timesteps,int(self.timesteps/2)):
                        if(np.sum(h5py_handle[videoIDs[j]+'/confidence'][k:k+self.timesteps,...]) > 50*(k+self.timesteps-k) ): ## Values of 100 or less seem to always mean there is no mouse in the frame
                            self.data.append(
                                (
                                    h5py_handle[videoIDs[j]+'/transformed_points'][k:k+self.timesteps,0:9,:].astype(np.float32), 
                                    h5py_handle[videoIDs[j]+'/points'][k:k+self.timesteps,:,:].astype(np.float32),
                                    self.labels[k,i]
                                )
                            )            
        else:
            for i in range(len(self.handle_list)):
                h5py_handle = self.handle_list[i]
                videoIDs = [i for i in h5py_handle]
                for j in range(len(videoIDs)):
                    for k in range(0,h5py_handle[videoIDs[j]+'/transformed_points'].shape[0]-self.timesteps,int(self.timesteps/2)):
                        if(np.sum(h5py_handle[videoIDs[j]+'/confidence'][k:k+self.timesteps,...]) > 50*(k+self.timesteps-k) ): ## Values of 100 or less seem to always mean there is no mouse in the frame
                            self.data.append(
                                (
                                    h5py_handle[videoIDs[j]+'/transformed_points'][k:k+self.timesteps,0:9,:].astype(np.float32),
                                    h5py_handle[videoIDs[j]+'/points'][k:k+self.timesteps,:,:].astype(np.float32),
                                    0
                                )
                            )
        self.num_samples = len(self.data)
    ############  Data loader and start workers is not strictly needed for my current implementation. Leaving in case I need it later
    def data_loader( self ):
        data_copy = self.data.copy()
        x =  np.zeros( ( self.batch_size, self.timesteps, self.width, self.height ), dtype=np.float32 )
        y =  np.zeros( ( self.batch_size, self.timesteps, self.width, self.height ), dtype=np.float32 )
        curr_index = 0
        while curr_index < self.num_samples-self.batch_size:
            for i in range(self.batch_size):
                x[i,...] = data_copy[curr_index][0]
                y[i,...] = data_copy[curr_index][1]
                curr_index += 1
            yield x, y
    def start_workers( self ):
        def worker(q,gen_func):
            x_ = np.zeros( ( self.batch_size, self.timesteps, self.width, self.height ), dtype=np.float32 )
            y_ = np.zeros( ( self.batch_size, self.timesteps, self.width, self.height ), dtype=np.float32 )
            iterate = True
            while iterate:
                try:
                    x_, y_ = next( gen_func )
                except:
                    logging.exception('Problem in loading from generator')
                    iterate = False
                try:
                    self.q.appendleft( ( x_, y_ ) )
                except:
                    logging.exception('Problem in adding to the queue')
                    iterate = False
        threads = []
        loaders = []
        for i in range( self.num_proc ):
            loaders.append( self.data_loader() )
            threads.append( threading.Thread( target = worker, args= ( self.q, loaders[i] ) ) )
            threads[i].setDaemon( True )
            threads[i].start()
'''
class mnist_Dataset():
    def __init__():
        from keras.datasets import mnist
'''