# Code for computing, data, models and Graphs for Paper 3.
#
#
# @ Alexander Tureczek, Technical University of Denmark, 2017
#
# Description: Cluster validation indices
#
# Implemented; Davies-Boudin, MIA, Cluster-Dispersion Index CDI.

import pandas as pd
import numpy as np

import clustering

#from line_profiler import LineProfiler

from scipy.spatial.distance import squareform, pdist


#For kernprof
#cd C:\OneDrive\Python\Paper 2\final
#kernprof -l -v cluster_ind_test.py



#se = pd.read_csv(path, sep=',', parse_dates=['dato'], dtype={'forbrug':np.float64}, nrows = 8000000)
#ava = pd.read_csv(path, sep=';', parse_dates=['date'], index_col='date')

'''
#prep data for pivot
#add hour to date var. 
se['idx'] = se.dato +pd.to_timedelta(se.hh, unit='h')

#set idx as index.
se = se.set_index(se['idx'])

#pivot making instnr columns for k-means
se_rdy = se.pivot(index='idx', columns='instnr', values='forbrug')
#print(se_rdy.head(4))

nans=se_rdy.isnull().any()
#only keep no-null
nans = nans[nans==False]
nani = nans.index
se_rdy = se_rdy[nani]
#print(se_rdy.head(4))

#print(se_rdy.shape)
heads = se_rdy.columns

data = se_rdy[heads[:1000]]
#data = se_rdy
#data = ava
print(data.shape)
[mean, model] = clustering.classify(data, 50, rs=12345, plotting= 'No')
'''

#----------------------------------------------
#          Davies-Boudin Index DBI
#----------------------------------------------
#@profile
def dbi_index(data, model):
    '''
    Calculation of the Davies-Boudin Index

    Syntak: [dbi, diam_max, ii, c_distance] = dbi_index(data, model)

    Input:

        data: pandas DataFrame with columns as the variables.

        model: model-object from "sklearn import cluster", or the classify function.

    Output:

        dbi: the Davies-Boudin Index

        diam_max: max diameter of cluster

        ii: counter

        c_distance: distance between clusters.


    '''
    ii= {}
    jj= {}
    diam_max={}
    dbi=0
    head = data.columns.values
    diam = pd.DataFrame()
    d_list = []
    dia_ij = {}

    #distance between each cluster and count
    c_center = model.cluster_centers_
    clusters = len(c_center)
    c_distance = pd.DataFrame(squareform(pdist(c_center)))

    # Diameter af cluster
    class_diam = {}
    class_diam_test = {}
    #class_diam = pd.DataFrame()
    for i in range(clusters):
        klasse = head[model.labels_==i]
        class_diam[i] =(data[klasse].max(axis=1)-data[klasse].min(axis=1)).mean()
        
    
    #Calculating top of DBI fraction, finding the largest sum of cluster diameters
    for i in class_diam:
        d_list = []
        for j in class_diam:
            if i == j:
                d_list.append(0)
            else:
                d_list.append(class_diam[i]+class_diam[j])

        diam[i] = d_list

    for i in diam.columns.values:
            dia_ij[i] =  diam[i][diam[i].idxmax()] / c_distance[i][diam[i].idxmax()]
	
    dbi = sum(list(dia_ij.values()))/clusters

    return dbi, diam_max, ii, c_distance

#----------------------------------------------
#          MIA & CDI index
#----------------------------------------------
#@profile
#----------------------------------------------
#          MIA & CDI index
#----------------------------------------------
def mia_index(data, model):
    '''
    Calculation of the MIA and CDI Index

    Syntak: [class_distance, mia, cdi] = mia_index(data, model)

    Input:

        data: pandas DataFrame with columns as the variables.

        model: model-object from "sklearn import cluster", or the classify function.

    Output:

        mia: the MIA Index

        cdi: the Cluster-Dispersion Index

        class_distance: distance between classes.


    '''

    class_distance=[]
    #head = data.columns.values
    mia = []

    #classes in model
    classes = model.labels_.view()
    #number of unique class labels
    unik = len(np.unique(classes))
    for i in range(unik):
        dist_2_center = 0
        dist_2center_sq = 0

        label = data.columns[model.labels_==i].values
        
        #meters = data[label].astype(float)
        meters = data[label]

        ##clct = pd.DataFrame(np.tile(model.cluster_centers_[i], len(meters)), index = meters.index, columns=label)
        clus_center = pd.DataFrame(np.tile(model.cluster_centers_[i], len(label)).reshape(len(label),data.shape[0]).transpose(), index = meters.index, columns=label)
        ##clct = pd.DataFrame(np.tile(model.cluster_centers_[i], len(label)).reshape(744,len(label)))

        #clus_center = pd.DataFrame(model.cluster_centers_[i])
        #Expanding cluster center to DF of same size as meters in the class. For subtraction purposes.
        #for i in range(meters.shape[1]-1):
        #    clus_center[i+1]=clus_center[0]

        
        #print('Difference:',str(clus_center.sum(axis=1).sum(axis=0) - clct.sum(axis=1).sum(axis=0)))
        #Setting header and index on cluster center series
        #clus_center.columns = meters.columns.values
        #clus_center = clus_center.set_index(meters.index)

        #Squared dist from each meter to center
        dist_2_center = meters.subtract(clus_center)
        dist_2_center_sq = dist_2_center**2

        #row average sq distance (axis=0)
        class_distance.append(dist_2_center_sq.mean(axis=0).mean())
        

        mia = np.sqrt(sum(class_distance)/unik)
        #From paper Structured Literature Review of Electricity Consumption Classification Using Smart Meter Data
        ##max distance between clusters
        #cdi = mia/clus_center.max().max()
        ## mean distance between clusters. 
        cdi = mia/clus_center.mean().mean()

        
    
    return class_distance, mia, cdi



#[dbi, diam_max, ii, c_distance] = dbi_index(data, model)
#[class_distance, mia, cdi] = mia_index(data, model)
