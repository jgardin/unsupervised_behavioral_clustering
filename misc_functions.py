##############################################################################################
### Sections of this code (the cluster assignment hardening loss calculation)
### were taken from or based on code in Elie Aljalbout's github
### https://github.com/elieJalbout/Clustering-with-Deep-learning/
###
### Direction for this approach was inspired by the associated paper
### https://arxiv.org/abs/1801.07648
###
###############################################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.animation as manimation
import cv2

#####################################
###  Losses
#####################################
def cluster_hardening_loss( y_true, y_predict ):
    f = tf.reduce_sum( y_predict, axis=0 )
    pij_numerator = y_predict * y_predict
    pij_numerator = pij_numerator / f
    normalizer_p = tf.reshape( tf.reduce_sum( pij_numerator, axis=1 ), ( -1, 1 ) )
    Pij = pij_numerator / normalizer_p
    log_arg = Pij / y_predict
    log_exp = tf.log( log_arg )
    sum_arg = Pij * log_exp
    return tf.reduce_sum( tf.reduce_sum( sum_arg, axis=1 ), axis=0 )


def getSoftAssignments( latent_space, cluster_centers, num_clusters, latent_space_dim, num_samples):
    '''
    Returns cluster membership distribution for each sample
    :param latent_space: latent space representation of current batch inputs [flattened mapping == batch_size * latent_space_dim]
    :param cluster_centers: the coordinates of cluster centers in latent space
    :param num_clusters: total number of clusters
    :param latent_space_dim: dimensionality of latent space [flattened size of the narrowest encoder layer, or whichever layer you choose to cluster]
    :param num_samples: total number of input samples  ==>  batchsize
    :return: soft assigment based on the equation qij = (1+|zi - uj|^2)^(-1)/sum_j'((1+|zi - uj'|^2)^(-1))
    ##### Example for the shapes of these variables
    num_samples = 16
    num_clusters = 40
    latent_space_dim = 64
    cluster_centers.shape ==> (40, 64)
    z_expanded.shape ==> (16, 40, 64)
    u_expanded.shape ==> (16, 40, 64)
    qij_numerator.shape ==> (16, 40)
    normalizer_q ==> (16, 1)
    RETURN ==> (16, 40)
    '''
    z_expanded = tf.reshape( latent_space, (num_samples, 1, latent_space_dim) )
    z_expanded = tf.tile( z_expanded, [1, num_clusters, 1] )
    u_expanded = tf.tile( tf.reshape( cluster_centers, ( 1, num_clusters, latent_space_dim ) ), [num_samples, 1, 1] )
    distances_from_cluster_centers = tf.norm( ( z_expanded - u_expanded ), axis=2, ord='euclidean' )
    qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
    qij_numerator = 1 / qij_numerator
    normalizer_q = tf.reshape( tf.reduce_sum( qij_numerator, axis=1 ), ( num_samples, 1 ) )
    return qij_numerator / normalizer_q



################################################################################################################################
###   Plotting functions
###
#################################################################################################################################

#syntaxLabels = ['NotGrooming','PawLicking','FaceWashing','FlankLicking','GenitalGrooming','TailGrooming','LegGrooming','Scratching','LickingRearPaw','UnilateralFaceWashing']
#xlabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']
def plot_label_matrix(cm, xlabel, ylabel,
                          normalize=False,
                          title='Annotated Behavior Across Clusters',
                          cmap=plt.cm.Blues,withNum=False):
    #### modified from scikit learn confusion matrix example #####
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xtick_marks = np.arange(len(xlabel))
    ytick_marks = np.arange(len(ylabel))
    plt.xticks(xtick_marks, xlabel, rotation=45)
    plt.yticks(ytick_marks, ylabel)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if(withNum):
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    return True

def tsne_clustering(data,labels,fig, ax, N, labelNames=None):
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000, learning_rate=200 )
    tsne_results = tsne.fit_transform(np.reshape(data,(data.shape[0],-1)))
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0,N,N+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    scat = ax.scatter(tsne_results[:,0],tsne_results[:,1],c=labels,cmap=cmap,norm=norm,s=2)
    cb = fig.colorbar(scat, spacing='proportional',ticks=bounds)
    if(labelNames is not None):
        cb.ax.set_yticklabels(labelNames)
    return True

def plot_ClusterPosture(parameters,syntacticDatasetObj,clusterIndex,fig,path):
    ##################################################################################################
    ### syntacticDatasetObj is the syntactic dataset class object from cluster_datasets.py
    ### clusterIndex is a list of all indices in syntacticDatasetObj.data for a cluster
    ### fig is the pyplot figure object to plot on
    ### ax1 is the average posture of all the frames in the cluster
    ### ax2 is the overlay of all postures in all frames of the cluster
    ##################################################################################################
    clusterSubset = np.stack([syntacticDatasetObj.data[i][0] for i in clusterIndex ],axis=0)
    clusterMedian = np.squeeze(np.median(clusterSubset,axis=0)) ##shape = 15,12,2
    ax1 = fig.add_subplot(1,1,1)
    xtext = ax1.set_xlabel('Posture Space X') 
    ytext = ax1.set_ylabel('Posture Space Y')
    def update(i):
        plt.ylim(clusterMedian[:,:,0].min(),clusterMedian[:,:,0].max())
        plt.xlim(clusterMedian[:,:,1].min(),clusterMedian[:,:,1].max())
        ### Head
        line = ax1.plot([clusterMedian[i,0,1],clusterMedian[i,1,1]], [clusterMedian[i,0,0],clusterMedian[i,1,0]], '-', color='purple', linewidth=1)
        line = ax1.plot([clusterMedian[i,0,1],clusterMedian[i,2,1]], [clusterMedian[i,0,0],clusterMedian[i,2,0]], '-', color='purple', linewidth=1)
        ### Spine + tail
        line = ax1.plot([clusterMedian[i,3,1],clusterMedian[i,6,1]], [clusterMedian[i,3,0],clusterMedian[i,6,0]], '-', color='red', linewidth=1)
#        line = ax1.plot([clusterMedian[i,6,1],clusterMedian[i,9,1]], [clusterMedian[i,6,0],clusterMedian[i,9,0]], '-', color='red', linewidth=1)
  #      line = ax1.plot([clusterMedian[i,9,1],clusterMedian[i,10,1]], [clusterMedian[i,9,0],clusterMedian[i,10,0]], '-', color='red', linewidth=1)
  #      line = ax1.plot([clusterMedian[i,10,1],clusterMedian[i,11,1]], [clusterMedian[i,10,0],clusterMedian[i,11,0]], '-', color='red', linewidth=1)
        ### Paws
        line = ax1.plot([clusterMedian[i,6,1],clusterMedian[i,4,1]], [clusterMedian[i,6,0],clusterMedian[i,4,0]], '-', color='green', linewidth=1)
 #       line = ax1.plot([clusterMedian[i,9,1],clusterMedian[i,7,1]], [clusterMedian[i,9,0],clusterMedian[i,7,0]], '-', color='green', linewidth=1)
        line = ax1.plot([clusterMedian[i,5,1],clusterMedian[i,6,1]], [clusterMedian[i,5,0],clusterMedian[i,6,0]], '-', color='blue', linewidth=1)
 #       line = ax1.plot([clusterMedian[i,8,1],clusterMedian[i,9,1]], [clusterMedian[i,8,0],clusterMedian[i,9,0]], '-', color='blue', linewidth=1)  
        line = ax1.plot([clusterMedian[i,8,1],clusterMedian[i,6,1]], [clusterMedian[i,8,0],clusterMedian[i,6,0]], '-', color='blue', linewidth=1)
        line = ax1.plot([clusterMedian[i,6,1],clusterMedian[i,7,1]], [clusterMedian[i,6,0],clusterMedian[i,7,0]], '-', color='green', linewidth=1)
        return line,ax1
    metadata = dict(title='Posture Cluster Animation', artist='Matplotlib', comment=str(path))
    writer = manimation.FFMpegWriter(fps=10, metadata=metadata)
    with writer.saving(fig, str(path), 96):
        for i in np.arange(0, parameters['timesteps']):
            update(i)
            writer.grab_frame()
            plt.cla()
#    writer.finish()
    return True

def inferClusterPlot( writer, index, max_index, points):
    BLUE = [255,0,0]
    RED = [0,0,255]
    GREEN = [0,255,0]
    BLACK = [0,0,0]
    WHITE = [255,255,255]
    ORANGE = [255,127,0]
    PURPLE = [106,61,154]
    YELLOW = [255,255,0]
    midFrame = int(points.shape[0]/2)
    new_img = np.zeros((480,512,3),dtype=np.uint8)
#######################################
    cv2.line(new_img, (int(points[midFrame,0,1]),int(points[midFrame,0,0])), (int(points[midFrame,3,1]),int(points[midFrame,3,0])), RED[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,3,1]),int(points[midFrame,3,0])), (int(points[midFrame,6,1]),int(points[midFrame,6,0])), RED[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,6,1]),int(points[midFrame,6,0])), (int(points[midFrame,9,1]),int(points[midFrame,9,0])), RED[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,9,1]),int(points[midFrame,9,0])), (int(points[midFrame,10,1]),int(points[midFrame,10,0])), RED[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,10,1]),int(points[midFrame,10,0])), (int(points[midFrame,11,1]),int(points[midFrame,11,0])), RED[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,3,1]),int(points[midFrame,3,0])), (int(points[midFrame,1,1]),int(points[midFrame,1,0])), YELLOW[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,3,1]),int(points[midFrame,3,0])), (int(points[midFrame,2,1]),int(points[midFrame,2,0])), GREEN[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,6,1]),int(points[midFrame,6,0])), (int(points[midFrame,4,1]),int(points[midFrame,4,0])), ORANGE[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,6,1]),int(points[midFrame,6,0])), (int(points[midFrame,5,1]),int(points[midFrame,5,0])), PURPLE[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,9,1]),int(points[midFrame,9,0])), (int(points[midFrame,7,1]),int(points[midFrame,7,0])), ORANGE[::-1], 1)
    cv2.line(new_img, (int(points[midFrame,9,1]),int(points[midFrame,9,0])), (int(points[midFrame,8,1]),int(points[midFrame,8,0])), PURPLE[::-1], 1)
    cluster_loc = int((480-1/max_index)*index/max_index)
    cv2.rectangle(new_img, (484,cluster_loc),(508,cluster_loc+int(1/max_index)), WHITE[::-1], 1)
    writer.append_data(new_img)

'''def plot_ClusterPosture(syntacticDatasetObj,clusterIndex,fig):
    ##################################################################################################
    ### syntacticDatasetObj is the syntactic dataset class object from cluster_datasets.py
    ### clusterIndex is a list of all indices in syntacticDatasetObj.data for a cluster
    ### fig is the pyplot figure object to plot on
    ### ax1 is the average posture of all the frames in the cluster
    ### ax2 is the overlay of all postures in all frames of the cluster
    ##################################################################################################
    clusterSubset = np.concatenate([syntacticDatasetObj.data[i][0] for i in clusterIndex ],axis=0)
    print('Clustersubset shape: {0!s}'.format(clusterSubset.shape)) ##5000*15,12,2
    #ax1,ax2 = fig.add_subplot(2,1,1)
    ax1 = fig.add_subplot(1,1,1)
    xtext = ax1.set_xlabel('Posture Space X') 
    ytext = ax1.set_ylabel('Posture Space Y')
    #xtext = ax2.set_xlabel('Posture Space X') 
    #ytext = ax2.set_ylabel('Posture Space Y')
    clusterMedian = np.median(clusterSubset,axis=0) ##shape = 12,2
    print('Cluster median shape: {0!s}'.format(clusterMedian.shape))
    ### Head
    line = ax1.plot([clusterMedian[0,1],clusterMedian[1,1]], [clusterMedian[0,0],clusterMedian[1,0]], '-', color='purple', linewidth=1)
    line = ax1.plot([clusterMedian[0,1],clusterMedian[2,1]], [clusterMedian[0,0],clusterMedian[2,0]], '-', color='purple', linewidth=1)
    ### Spine + tail
    line = ax1.plot([clusterMedian[3,1],clusterMedian[6,1]], [clusterMedian[3,0],clusterMedian[6,0]], '-', color='red', linewidth=1)
    line = ax1.plot([clusterMedian[6,1],clusterMedian[9,1]], [clusterMedian[6,0],clusterMedian[9,0]], '-', color='red', linewidth=1)
    line = ax1.plot([clusterMedian[9,1],clusterMedian[10,1]], [clusterMedian[9,0],clusterMedian[10,0]], '-', color='red', linewidth=1)
    line = ax1.plot([clusterMedian[10,1],clusterMedian[11,1]], [clusterMedian[10,0],clusterMedian[11,0]], '-', color='red', linewidth=1)
    ### Paws
    line = ax1.plot([clusterMedian[6,1],clusterMedian[4,1]], [clusterMedian[6,0],clusterMedian[4,0]], '-', color='green', linewidth=1)
    line = ax1.plot([clusterMedian[9,1],clusterMedian[7,1]], [clusterMedian[9,0],clusterMedian[7,0]], '-', color='green', linewidth=1)
    line = ax1.plot([clusterMedian[5,1],clusterMedian[6,1]], [clusterMedian[5,0],clusterMedian[6,0]], '-', color='blue', linewidth=1)
    line = ax1.plot([clusterMedian[8,1],clusterMedian[9,1]], [clusterMedian[8,0],clusterMedian[9,0]], '-', color='blue', linewidth=1)  
    return True
'''

'''   Old code
def model_infer_syntactic( parameters, modelpath, outputpathprefix, trainLoad, validLoad ):
    ##################################################
    ### Import pretrained model
    ### '/home/gardij/Code/NN_models/PointClustering_CAH_epoch_1425_18-02-18-10-09-42.h5'
    ##################################################
    parameters['training_size'] = int( trainLoad.num_samples )
    parameters['validation_size'] = int( validLoad.num_samples )
    print('Total training samples: {0!s}\nTotal validation samples: {1!s}'.format(parameters['training_size'],parameters['validation_size'] ))
    parameters['training_size'] = int( parameters['training_size'] / parameters['batch_size'] )
    parameters['validation_size'] = int( parameters['validation_size'] / parameters['batch_size'] )
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
    elif( parameters['model_type'] == 'singleFrame' ):
        input_, latent_, cluster_, decoder_ = cluster_models.encoder_decoder_singleFrame(
                                                    parameters['height'],
                                                    parameters['width'],
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
    else:
        print('Model type argument is required. Did you define your own model? If so, modify network_routines.py. Current Model type is: {0!s}'.format(parameters['model_type']))
        quit()
    Net_ = keras.models.Model(
                inputs = [ input_ ],
                outputs = [ latent_ ]
                )
    try:
        Net_.load_weights( modelpath,by_name=True)
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
    truth_labels = []
    ###This currently works, but is slow
    for element in trainLoad.data:
        latentVector = Net_.predict( np.expand_dims(element[0],axis=0), batch_size = 1)
        truth_labels.append(np.argmax(element[1]))
        initial_latent_space.append( np.reshape(latentVector,(parameters['latent_space_dim']),) )
    print( 'Starting Kmeans.' )
    latent_space_out = np.stack(initial_latent_space,axis=0)
    truth_labels = np.stack(truth_labels,axis=0)
    np.savetxt(str(outputpathprefix)+'_latent_space.csv', latent_space_out, delimiter = ',')
    kmeans_func = KMeans( n_clusters = parameters[ 'num_clusters' ], random_state = 0)
    kmeans_out = kmeans_func.fit( latent_space_out )  ### outputs sklearn kmeans object
    np.savetxt(str(outputpathprefix)+'_kmeans_centroids.csv', kmeans_out.cluster_centers_, delimiter = ',') ### shape ==> (num_clusters, dimension of latent space)
    np.savetxt(str(outputpathprefix)+'_kmeans_labels.csv', kmeans_out.labels_, delimiter = ',')
    xLabels = ["Cluster {0!s}".format(x) for x in range(0,parameters['num_clusters'])]
    syntaxLabels = ['NotGrooming','PawLicking','FaceWashing','FlankLicking','GenitalGrooming','TailGrooming','LegGrooming','Scratching','LickingRearPaw','UnilateralFaceWashing']
    cnf_matrix = confusion_matrix( kmeans_out.labels_, truth_labels )
    plt.figure(figsize=(20,12))
    misc_functions.plot_label_matrix(cnf_matrix[:,0:10].T,xlabel=xLabels,ylabel=syntaxLabels,normalize=True)
    plt.savefig(str(outputpathprefix)+'all_syntactic_kmeans_ConfusionMat_rowNormalize'+str(parameters['num_clusters'])+'.png')
    plt.figure(figsize=(20,12))
    misc_functions.plot_label_matrix(cnf_matrix[:,0:10],xlabel=syntaxLabels,ylabel=xLabels,normalize=True)
    plt.savefig(str(outputpathprefix)+'all_syntactic_kmeans_ConfusionMat_columnNormalize'+str(parameters['num_clusters'])+'.png')
    #plot_ClusterPosture(syntacticDatasetObj,clusterIndex,fig)
    def find(lst, a):
        return [i for i, x in enumerate(lst) if x == a]
    for i in range(parameters['num_clusters']):
        fig = plt.figure(figsize=(20,12))
        misc_functions.plot_ClusterPosture(trainLoad,
                            find(kmeans_out.labels_,i),
                            fig
                            )
        plt.savefig(str(outputpathprefix)+'all_syntactic_averagePosture_cluster'+str(i)+'.png')
        plt.close()
    print( 'Finished inference!' )
    '''
'''
def calculateP( Q ):
#    Function to calculate the desired distribution Q^2, for more details refer to DEC paper
#    Example shapes:
#    Q.shape ==> ( 16, 40 )
#    f.shape ==> ( 40, )
#    pij_numerator ==> ( 16, 40 )
#    num_samples = 16
#    num_clusters = 40
#    latent_space_dim = 64

    f = tf.reduce_sum( Q, axis=0 )
    pij_numerator = Q * Q
    pij_numerator = pij_numerator / f
    normalizer_p = tf.reshape( tf.reduce_sum( pij_numerator, axis=1 ), ( -1, 1 ) )
    P = pij_numerator / normalizer_p
    # print('P: {0!s}'.format(P))
    # P = tf.Print(P,[P,f,normalizer_p], message='P')
    return P

def getKLDivLossExpression( Q_expression, P_expression ):
    # Loss = KL Divergence between the two distributions
    log_arg = P_expression / Q_expression
    log_exp = tf.log( log_arg )
    sum_arg = P_expression * log_exp
    loss = tf.reduce_sum( tf.reduce_sum( sum_arg, axis=1 ), axis=0 )
    #print('KL Loss: {0!s}'.format(loss))
    #loss = tf.Print(loss,[loss,Q_expression], message='kl loss')
    return loss
'''