# Code for computing, data, models and Graphs for Paper 3.
#
#
# @ Alexander Tureczek, Technical University of Denmark, 2017
#
# Description: Ploting functions for cluster analysis.
import matplotlib.pyplot as plt
import collections
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np


#---------------------------------------------------
#
#      Plot indices development           
#
#---------------------------------------------------
def plt_idx(sil, mia, cdi, dbi, titel = 'Unknown dataset', size=30):
    '''
    Plotting function for plotting cluster validation index development as function of clusters.

    Syntax: plt_idx(sil, mia, cdi, dbi)

    INPUT:
        titel: title on graph, default = "Unknown dataset"

        sil, mia, cdi, dbi: dicts with the indices as named. 



    Output:

        Print of indices development
    '''
    y = list(sil.keys())

    fig  = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y, list(sil.values()), label='Silhouette')
    ax.plot(y, list(mia.values()), label='MIA')#, marker= 'o')
    ax.plot(y, list(cdi.values()), label='CDI')
    ax.plot(y, list(dbi.values()), label='DBI')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(titel, fontsize=size)
    plt.xlabel('# of Clusters', fontsize=size*(5/6))
    plt.ylabel('Index Value', fontsize=size*(5/6))
    plt.legend(fontsize=size*(4/6))
    plt.tick_params(axis='both', which='major', labelsize=size/2)
    plt.grid()
    plt.show(block=False)


#---------------------------------------------------
#           Plot of mean group plus obs.           
#---------------------------------------------------
def class_mean_plot(model, data, mean, lw=0.5, threshold =4, clm='nipy_spectral', size=60):
    """
    Function for plotting classes and class meters for all classes with more members than the threshold value.

    SYNTAX: class_mean_plot(model, data, mean, lw=0.05, threshold =4)

    USE: [mean, model] = classify(df_rdy,10)

    INPUT:

        model: is a model object from the sklearn.Cluster KMeans class.
                this model object can be obtained from the classify function [mean, model]

        data: the data used for classification.

        mean: is a class mean, this mean can be obtained from the classify function [mean, model]

        lw: linewidth of the class-members.

        OUTPUT: figure. 
    
    """

    fig = plt.figure(figsize=(8, 6))
        
    maxi = 6#model.labels_.max()
    head = data.columns.values
    key = collections.Counter(model.labels_) 

    #color
    cmap = plt.get_cmap(clm)
    
    for i in list(range(np.unique(model.labels_).max()+1)):#np.unique(model.labels_).sort():
        if len(model.labels_[model.labels_==i]) > threshold:
            st = 'group '+str(i)+' Obs: '+str(key[i])#+' obs'
            #color = cmap(float(i)/maxi)
            
            #if model.labels_[i] == 4:#i ==40:
                
                #-------------------------------- HARD CODED -----------------------------
                #plt.plot(data[head[model.labels_==4]], linewidth=lw, c='orange')
            #    plt.plot(mean[4], c='orange', marker='o', linewidth=2, label=st)
            #else: 
            plt.plot(data[head[model.labels_==i]], linewidth=lw)#, c=color)
            #    plt.plot(mean[i], c=color, linewidth=1, label=st, marker='o')

    

    titel = 'Classification Fit. Random Seed: 12345'#+str(rs)
    plt.title(titel, fontsize = size)
    plt.tick_params(axis='both', which='major', labelsize=str(size*(2/5)))
    plt.xlabel('Lag', fontsize=size*(5/6))
    plt.ylabel('Value', fontsize=size*(5/6))
    #plt.ylim(0,1)
    
    plt.grid()
    plt.legend(loc='upper right', fontsize=size*(2/5))
    plt.show(block=False)



#---------------------------------------------------
#
#      Plot from classification to original data           
#
#---------------------------------------------------
def plots():
    key = collections.Counter(model.labels_) 
    for i in range(4):
    
            color = cmap(float(i)/maxi)
            plt.plot(ava[ava.columns[model.labels_==i]], linewidth=0.75, c=color)#, label=st)

            size=30

    a = ['102', '114','101','111R']
    for i,j in zip(a, range(4)):
        st = 'group '+str(j)+' Obs: '+str(key[j])
        color = cmap(float(j)/maxi)
        plt.plot(ava[i], linewidth=0.75, c=color, label=st)



    titel = 'Class Mapping to Original Data'
    plt.title(titel, fontsize = size)
    plt.tick_params(axis='both', which='major', labelsize=str(size*(2/5)))
    plt.xlabel('Lag', fontsize=size*(5/6))
    plt.ylabel('Value', fontsize=size*(5/6))
    #plt.ylim(0,1)
        
    plt.grid()
    plt.legend(loc='upper right', fontsize=size*(2/3))
    plt.show(block=False)




#---------------------------------------------------
#
#           Plot single size clusters.           
#
#---------------------------------------------------
def single_class(model, data, save_plot=0):
    """
        Plotting classes with only 1 member.

        SYNTAX: single_class(model, data, save_plot=0)

        INPUT:

            model:

            data: the data used for modeling

            save_plots = save the plot yes = 1

            save to hardcoded location.
    """
    classes = model.labels_.view()
    head = data.columns.values
    unik = len(np.unique(classes))
    for i in range(unik):
            print('Class #',i, 'Size:', len(classes[classes==i]))

    fig2 = plt.figure()
    fig2.set_size_inches(18,12)
    j = 1
    tmp = []
    for i in range(unik):
            if len(classes[classes==i])==1:
                ax = fig2.add_subplot(4,3,j)
                txt = 'Class number: '+str(head[(model.labels_==i)])
                ax.title.set_text(txt)
                ax.plot(data[head[(model.labels_==i)]])
                j+=1
    
    if save_plot==1:
        plt.savefig('C:/Users/atur/Desktop/OneDrive/PhD/Writing/2 SE Classification/single_class.png')

    
    plt.show(block=False)


#----------------------------------------------
#                    Class plot
#----------------------------------------------
def classes(data, model, class_1=1, class_2=5, lw = 0.1, mlw=2, mean_only = 0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ee=data.isnull().any()
    ff = ee[ee==False].index
    head = data[ff].columns.values
    aa = data[ff][head[model.labels_==class_1]]
    b = data[ff][head[model.labels_==class_2]]

    if mean_only == 0:
        plt.plot(aa, 'blue', linewidth=lw)
        plt.plot(b, 'red', linewidth=lw)

    plt.plot(aa.mean(axis=1), 'blue', linewidth=mlw)
    plt.plot(b.mean(axis=1), 'red', linewidth=mlw)
    plt.title('2 classes, fat line is class mean')

    plt.title('Class means from 2 largest classes from K-means, normalized data, (Monday-Sunday)')
    plt.xlabel('Date time: 24h between each date')
    plt.ylabel('Consumption kW')
    plt.grid()
    plt.ylim(0,1)
    
    plt.show(block=False)


#



#----------------------------------------------
#          Plots transforms
#----------------------------------------------
def plot_transforms(data):
    
    [mc, norm, std, mdiv] = normalize(data)

    #Sorting by series mean to get nice color distribution in the plots. 
    avg = ava.mean(axis=0)
    avg2 = avg.sort_values()
    
    cmap = plt.get_cmap('Spectral') 

    
    #Original data
    plt.figure()
    for i in range(len(avg2.index)):
        color=cmap(i/len(avg2.index))
        plt.plot(ava[avg2.index[i]], c=color)  

    plt.title('Original Recorded Data', fontsize='30')
    plt.ylabel('Mw', fontsize=' 25')
    plt.xlabel('Date', fontsize='25')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid()
    plt.show(block=False)

    #Normalized data
    plt.figure()
    for i in range(len(avg2.index)):
        color=cmap(i/len(avg2.index))
        plt.plot(norm[avg2.index[i]], c=color)  

    plt.title('Normalized Data', fontsize='60')
    plt.ylabel('Mw', fontsize='50')
    plt.xlabel('Date', fontsize='50')
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.grid()
    plt.show(block=False)

    #Standardized data
    plt.figure()
    for i in range(len(avg2.index)):
        color=cmap(i/len(avg2.index))
        plt.plot(std[avg2.index[i]], c=color)  

    plt.title('Standardized Data', fontsize='60')
    plt.ylabel('Mw', fontsize='50')
    plt.xlabel('Date', fontsize='50')
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.grid()
    plt.show(block=False)

    #Mean-Centered data
    plt.figure()
    for i in range(len(avg2.index)):
        color=cmap(i/len(avg2.index))
        plt.plot(mc[avg2.index[i]], c=color)  

    plt.title('Mean-Centered Data', fontsize='60')
    plt.ylabel('Mw', fontsize='50')
    plt.xlabel('Date', fontsize='50')
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.grid()
    plt.show(block=False)

    #Mean-Divide data
    plt.figure()
    for i in range(len(avg2.index)):
        color=cmap(i/len(avg2.index))
        plt.plot(mdiv[avg2.index[i]], c=color)  

    plt.title('Mean-Divided Data', fontsize='60')
    plt.ylabel('Mw', fontsize='50')
    plt.xlabel('Date', fontsize='50')
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.grid()
    plt.show(block=False)



def plt_wav(original, wavelet):
    plt.figure()

    plt.plot(original, 'r')
    plt.plot(wavelet, 'b')

    plt.show(block=False)

    
    

