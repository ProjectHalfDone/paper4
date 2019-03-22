# Code for computing, data, models and Graphs for Paper 3.
#
#
# @ Alexander Tureczek, Technical University of Denmark, 2017
#
# Description: functions for clustering

from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections



#------------------------------------------------------
#                 Classification Part
#------------------------------------------------------
#def classify(data_rdy, clusters, threshold = 5):
def classify(data, clusters, model=None, rs=None, threshold = 4, plotting = 'Yes', titel = 'Unknown Data', size=30):
    ''' Classification method using sklearn KMeans
    if plotting = 'Yes' the method plots the class means above the threshold

    return mean {}, model sklearn KMeans object

    Apply to cluster via K-means with cluster number = to clusters input.
    '''

    if plotting == 'Yes':
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        fig = plt.figure()
        ax2 = fig.add_subplot(111)

    #If no model is given, calculate model using KMeans
    if model == None: 
        test = data.transpose()
        model = cluster.KMeans(n_clusters=clusters, random_state=rs).fit(test)#, verbose=True)

    #color
    cmap = plt.get_cmap('nipy_spectral')
    #maxi = len(np.unique(model.labels_))
    maxi = 6# model.labels_.max()

    'Initializing support variables'
    head = data.columns.values #Header
    mean = {} #mean
    key = collections.Counter(model.labels_) #

    lst = list(np.unique(model.labels_))
    
    for i in list(range(np.unique(model.labels_).max()+1)):
            if len(model.labels_[model.labels_==i]) > threshold:
                st = 'group '+str(i)+' Obs: '+str(key[i])#+' obs'
                mean[i] = data[head[(model.labels_==i)]].mean(axis=1)
                x = list(range(data.shape[0]))
                if plotting == 'Yes':
                    if i == 40:
                        #-------------------------------- HARD CODED -----------------------------
                        plt.plot(mean[4], c='orange', linewidth=2, label=st)
                    else:
                        col = cmap(float(i)/maxi)
                        ax1.scatter(x, mean[i], label = st, color = col, linewidth=2)
                        ax2.plot(x, mean[i], label = st, color = col, linewidth=2)
                        #plt.scatter(x, mean[i], label = st, color = col, linewidth=2)#, marker='o')
                        #plt.plot(x, mean[i], label = st, color = col, linewidth=2)#, marker='o')

    if plotting == 'Yes':
        plt.figure()
        print('classification')
        print(model.labels_)
        print(model.inertia_)
        #titel = titel + ', Random seed: '+str(rs)
        titel = titel + ', RS: '+str(rs)
        plt.title(titel, fontsize = size)
        plt.tick_params(axis='both', which='major', labelsize=str(size*(2/5)))
        plt.xlabel('Lag', fontsize=size*(5/6))
        plt.ylabel('Value', fontsize=size*(5/6))
        #plt.ylim(0,1)
        
        plt.grid()
        plt.legend(loc='upper right', fontsize=size*(2/5))
        plt.show(block=False)

    return mean, model

    #[mean, model] = classify(data, i, rs=rs, plotting= 'No')
    #[mean, model] = classify(mdiv, 4, rs=rs, plotting= 'Yes')


