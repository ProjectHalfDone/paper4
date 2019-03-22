# Code for computing, data, models and Graphs for Paper 3.
#
#
# @ Alexander Tureczek, Technical University of Denmark, 2017
#
#


#------------INITIALIZE------------INITIALIZE------------INITIALIZE------------INITIALIZE------------INITIALIZE 



#------------------------------------------------------
#           Import Modules
#------------------------------------------------------
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import sklearn.decomposition as dcmp
import xlrd

import time

from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.spatial.distance import squareform, pdist
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from statsmodels.robust import mad
print('Error msg: statsmodels.tsa.stattools import')
from statsmodels.tsa.stattools import acf, pacf


#Own functions
import auxil
import coef_pyramid_plot as cpp
import clustering
import cluster_indices  
import ploting
import wavelets



#------------IMPORT DATA------------IMPORT DATA------------IMPORT DATA------------IMPORT DATA------------IMPORT DATA

#------------------------------------------------------
#           Import data: SE data
#------------------------------------------------------

#Data file removed...!


#remove week 52 as it does not fit with an entire week.
se = se_init[se_init.week!=52]


#Check that SE data is imported correct
print('Shape of january, expect (87600, 26563) obs',se.shape) 

#check for null values
print('Missing values in data:', se.isnull().values.any())

#week 2 of january 2011 - 10th to 16th
#test = se_rdy.ix['2011-01-10':'2011-01-16']


#q1 = se[se['week'].isin(list(range(1,14)))]

#------------Preprocessing------------Preprocessing------------Preprocessing------------Preprocessing------------Preprocessing


#------------------------------------------------------
#           Scale Data:  Normalizing
#                        Mean-Center
#                        Mean-Divide
#                        Standardize
#
#------------------------------------------------------
def scale(data):
    ''' Method for scaling data, removing difference in unit and alligning data.
    
    Syntax: [mc, norm, std, mdiv] = scale(data)

    Input: Data, pandas DataFrame, where columns are to be scaled.

    Output:

        mc: mean-centered data, essentially setting mean = 0.

        norm: normalizing, mapping data to interval [0-1] smallest value in column is set to 0 largest to 1.

        std: standardizing data, relating all observations to data standard deviation.

        mdiv: Mean-divide data, ...
    
    '''
    #making a copy of data for the mean centering.
    norm = data.copy()
    std = data.copy()
    mdiv = data.copy()

    #Normalize
    norm = (norm - norm.min(axis=0)) / (norm.max(axis=0) - norm.min(axis=0))
    
    std = (std - std.mean(axis=0)) / (std.std(axis=0))

    #mean division
    mdiv = (mdiv)/(mdiv.mean(axis=0))

    #making a copy of data for the mean centering.
    mc = data.copy()

    for i in mc.columns.values:
        mc[i] = mc[i]-mc[i].mean()
        
    return mc, norm, std, mdiv

    #[mc, norm, std, mdiv] = scale(se)






#------------------------------------------------------
#
#       Autocorrelation Calculation prep
#   
#------------------------------------------------------
def acf_prep(data, lags = 24, correction='Yes'):
    """ Calculate Autocorrelation funciton for each of the input colums. 

    Syntax: acf_prep(data, lags = 24, correction='Yes'):

    INPUT:

        data : data to analyze, pandas DataFrame ACF calculated on columns.

        lags : number of lags to calculated

        correction : removing non-significant coefficients. default = 'Yes'

    OUTPUT:

        acf : autocorrelation function for each column

    """
    
    aacf = pd.DataFrame()

    for i in data.columns:
        [auto, conf, qstat, pvalues] = acf(data[i], qstat=True, alpha=0.05, nlags=lags)

        if correction =='Yes':
            if conf.min()<=0:
                auto[conf[:,0]<=0]=0
        
        aacf[i] = auto
    
        
    return aacf

    #w1_acf = acf_prep(data, lags = 24, correction='Yes')



#------------------------------------------------------
#
#       Autocorrelation Calculation PLOT
#   
#------------------------------------------------------
def acf_plot(meter):
    '''
    Requires the dataset se from import data to be active.

    good meters
    '344113530'
    '344116979'

    '344109496'
    
    '''
    size = 30
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    [auto, conf, qstat, pvalues] = acf(se[meter], qstat=True, alpha=0.05, nlags=48)

    ax1.plot(se[meter])
    ax1.set_title('Original Consumption', fontsize=size*(5/6))
    
    plt.sca(ax1)
    plt.xticks(rotation=45)
    
    ax2.plot(auto, color = 'b', linewidth=2)
    ax2.plot(conf[:,0], color = 'b', linestyle = '--', linewidth = 1)
    ax2.plot(conf[:,1], color = 'b', linestyle = '--', linewidth = 1)
    ax2.axhline(0, color = 'k')
    ax2.set_title('Series Autocorrelation', fontsize=size*(5/6))

    #significance correction
    if conf.min()<=0: # remove if
            auto[(conf[:,0]<=0) & (conf[:,1]>=0)] = 0 
    
    ax3.plot(auto)
    ax3.set_title('Retained Significant Features', fontsize=size*(5/6))

    ax2.set_xlabel('Lag', fontsize=size*(5/6))
    ax3.set_xlabel('Lag', fontsize=size*(5/6))
    
    ax1.set_ylabel('Consumption kWh', fontsize=size*(5/6))
    ax1.tick_params(axis='both', which='major', labelsize=size/2)
    ax2.tick_params(axis='both', which='major', labelsize=size/2)
    ax3.tick_params(axis='both', which='major', labelsize=size/2)

    
    plt.show(block=False)
    
    

#------------Classify------------Classify------------Classify------------Classify------------Classify------------Classify


#----------------------------------------------
#          KMeans + SIL, MIA & Inertia
#----------------------------------------------

def clusters(data, max_cluster=10, rs=None):#, titel = 'Unknown dataset',size = 30):
    ''' Clustering with different number of clusters.

    Syntax:

        [sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(data, max_clusters = 10, rs = None)

    INPUT: Pandas dataframe compliant with Sklearn KMeans classification.

    data: data to classify, pandas DataFrame
    max_cluster: number of clusters to evaluate. this is the upper bound, så max_cluster=10 gives
    rs: random_state integer value, if selected trials can be repeated
                clustering with all combinations from 2-10 clusters, and calculates Inertia, CDI, DBI, MIA, Silhouette

    output: all given as dicts
    sil - Silhouette index 
    mia - MIA index
    CDI - CDI index
    DBI - Davies-Bouldin index
    inerti - inertia calculated by SKlearn KMeans method.
    models - all models evaluated.
    c_pct: percent of classes with more than 1 member.


    ex.
    [sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(ava, 20, 12345)
    '''

    sil = {}
    mia={}
    cdi = {}
    dbi={}
    inerti = {}
    models = {}
    c_len = []
    c_pct = {}

    diam_max = {}
    ii = {}
    c_distance = {}
    
    
    for i in range(2,max_cluster+1):
        #Classify data
        start = time.time()
        print('Cluster number:',str(i), 'of', str(max_cluster),'clusters')
        [mean, model] = clustering.classify(data, i, rs=rs, plotting= 'No')
        clus = time.time()
        #print('Clustering: ',str(i),' time taken:', str(clus-start))
        
        #Calculate index
        [dbis, diam_max, ii, c_distance] = cluster_indices.dbi_index(data, model)
        dbi_time = time.time()
        sil[i]= silhouette_score(data.transpose(), model.labels_)
        #print('DBI Index: ',str(i),' time taken:', str(dbi_time-clus))
        
        #sil[i]= cluster_indices.silhouette_score(data.transpose(), model.labels_)
        [class_dist, mias, cdis] = cluster_indices.mia_index(data, model)
        mia_time = time.time()
        #print('MIA Index: ',str(i),' time taken:', str(mia_time-dbi_time))
    
        #Model inertia
        inerti[i] = model.inertia_

        #Calculate percent of small clusters.
        #Cluster labels and unique labels
        classes = model.labels_.view()
        unik = len(np.unique(classes))
        for j in range(unik):
            c_len.append(len(classes[classes==j]))

        #pct of classes with more than 1 member
        c_len_np = np.array(c_len)
        c_pct[i] = sum(c_len_np>1)/len(c_len_np)

        models[i] = model
        mia[i] = mias
        cdi[i] = cdis
        dbi[i] = dbis
        #print('Cluster #:', str(i))

    end = time.time()-start
    #print('total time:', str(end), 'seconds, eq. to ',str(end/60),'min')

    #remove if silhouette score is calculated
    for i in range(2,max_cluster+1):
        sil[i] = 0 
    
    return sil, mia, cdi, dbi, inerti, models, c_pct


    
#>>> mc = mean_center(df_rdy)
#>>> [sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(norm, 3, 12345)    
#>>> [sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(norm, 20, rs=12345, titel ='Normalized')





#------------Wavelets------------Wavelets------------Wavelets------------Wavelets------------Wavelets------------Wavelets------------Wavelets

         

#----------------------------------------------
#          Convert data to wavelet
#          coefficients for classification 
#----------------------------------------------


def wav_run(data, wavelet='coif16', mode='hard'):
    
    #Initialize
    #dn = []
    w_coef = pd.DataFrame()
    
    #Iterating over all columns in data. 
    for i in data.columns:
        dn = []
        [true_coefs, signal, denoised] = wavelets.wave(data[i], wavelet=wavelet, mode=mode)

        #first levels of coefficients shall not be included
        #Converting dict to list.
        for j in range(len(denoised)-1):
            dn = dn + list(denoised[j+1])
                
        w_coef[i] = dn

    return w_coef, dn
        






'''
[mc, norm, std, mdiv] = scale(ava)
[wav, dn] = wav_run(mc)
[sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(wav, 20, rs=12345)
ploting.plt_idx(sil, mia, cdi, dbi, titel = 'Mean-Centered Wavelet, RS = 12345', size = 60)

[mc, norm, std, mdiv] = scale(ava)
[wav, dn] = wav_run(norm)
[sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(wav, 20, rs=12345)
ploting.plt_idx(sil, mia, cdi, dbi, titel = 'Normalized Wavelet, RS = 12345', size = 60)

[mc, norm, std, mdiv] = scale(ava)
[wav, dn] = wav_run(std)
[sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(wav, 20, rs=12345)
ploting.plt_idx(sil, mia, cdi, dbi, titel = 'Standardized Wavelet, RS = 12345', size = 60)

[mc, norm, std, mdiv] = scale(ava)
[wav, dn] = wav_run(mdiv)
[sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(wav, 20, rs=12345)
ploting.plt_idx(sil, mia, cdi, dbi, titel = 'Mean-Divided Wavelet, RS = 12345', size = 60)
'''
'''
[wav, dn] = wav_run(ava)
[sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(wav, 20, rs=12345)

for i in range(5):
    print('-----**',str(i+1),'**-----')
    auxil.cls_dst(models[i+1])    
'''


#ploting.plt_idx(sil, mia, cdi, dbi, titel = 'Original Data Wavelet, RS = 12345', size = 60)
#[mean, model] = clustering.classify(wav, 4, rs=12345, threshold=0, plotting= 'Yes')
#auxil.cls_dst(model)


def nfold(lst, splits):

    '''
    nfold is a method for splitting a list into n splits for cross-validation.
    it takes as input a list randomizes the list and removes n equally sized sublists from the list like n fold cross validation
    the output is dictionary with n different sub lists ready for input for clustering. 

    Syntax:

    final_list = nfold(lst, splits)
    
    input:
        lst = list to be split into sublists

        splits = number of splits to perform

    output:
        final_list = dictionary with all sublists ready for clustering. 


    '''

    #initialize sets
    rlst =  np.random.permutation(lst)
    lst = set(lst)
    size = len(rlst)

    #initialize variables
    final_list = {}
    pcs = int(round(size/splits, 0))
    start = 0

    #create sublists using sets. created by moving a window across the list. 
    for i in range(splits):
        end = pcs+start
        drop = set(rlst[start:end])
        
        start = end
        final_list[i] = lst.difference(drop)
        #print(drop, 'Remainder',lst.difference(drop))
        #print(drop, 'Remainder',final_list[i])
    return final_list






#normalize data.
#[mc, norm, std, mdiv] = scale(se)



def cv(data, max_clusters = 3, rs = 12345, size = 60, titel = 'Unknown Data'):
    ''''leave-one-out cross validation
    data easily generates memory error.

    use cv_nfold in stead. 
    '''
    
    looc = {}
    #Create dictionary values for leave-one-out crossvalidation.looc = {}
    for i in data.columns:
            looc[i] = [col for col in data.columns if col not in i]

    silhouette = pd.DataFrame()
    mia_ind = pd.DataFrame()
    cluster_db = pd.DataFrame()
    davies = pd.DataFrame()
   

    for i in looc.keys():
        [sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(data[looc[str(i)]], max_clusters, rs=rs)



    #return looc

        silhouette[i] = sil.values()
        mia_ind[i] = mia.values()
        cluster_db[i] = cdi.values()
        davies[i] = dbi.values()

    
    
    x = list(range(2,max_clusters,1))

    sil_pm = pd.DataFrame()
    mia_pm = pd.DataFrame()
    cdi_pm = pd.DataFrame()
    dbi_pm = pd.DataFrame()
         
    sil_pm['Upper'] = silhouette.max(axis=1)
    sil_pm['Lower'] = silhouette.min(axis=1)
    sil_pm['Mean'] = silhouette.mean(axis=1)

    mia_pm['Upper'] = mia_ind.max(axis=1)
    mia_pm['Lower'] = mia_ind.min(axis=1)
    mia_pm['Mean'] = mia_ind.mean(axis=1)

    cdi_pm['Upper'] = cluster_db.max(axis=1)
    cdi_pm['Lower'] = cluster_db.min(axis=1)
    cdi_pm['Mean'] = cluster_db.mean(axis=1)

    dbi_pm['Upper'] = davies.max(axis=1)
    dbi_pm['Lower'] = davies.min(axis=1)
    dbi_pm['Mean'] = davies.mean(axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot(x, sil_pm['Mean'], 'b', linewidth=3, label = 'Silhouette Mean')
    ax.plot(x, dbi_pm['Mean'], 'r', linewidth=3, label = 'DBI Mean')
    ax.plot(x, mia_pm['Mean'], 'k', linewidth=3, label = 'MIA Mean')
    ax.plot(x, cdi_pm['Mean'], 'g', linewidth=3, label = 'CDI Mean')
             
    ax.plot(x, sil_pm['Lower'], 'b--', linewidth=1, label='_nolegend_')
    ax.plot(x, sil_pm['Upper'], 'b--', linewidth=1, label='_nolegend_')

    ax.plot(x, mia_pm['Lower'], 'k--', linewidth=1, label='_nolegend_')
    ax.plot(x, mia_pm['Upper'], 'k--', linewidth=1, label='_nolegend_')

    ax.plot(x, cdi_pm['Lower'], 'g--', linewidth=1, label='_nolegend_')
    ax.plot(x, cdi_pm['Upper'], 'g--', linewidth=1, label='_nolegend_')

    ax.plot(x, dbi_pm['Lower'], 'r--', linewidth=1, label='_nolegend_')
    ax.plot(x, dbi_pm['Upper'], 'r--', linewidth=1, label='_nolegend_')

    plt.legend(loc='upper right')

    plt.title(titel, fontsize=size)
    plt.xlabel('# of Clusters', fontsize=size*(5/6))
    plt.ylabel('Index Value', fontsize=size*(5/6))
    plt.legend(fontsize=size*(4/6))
    plt.tick_params(axis='both', which='major', labelsize=size/2)
    plt.grid()

    plt.show(block=False)

    return silhouette, mia_ind, cluster_db, davies

    #[silhouette, mia_ind, cluster_db, davies] = cv(data, max_clusters = 3, rs = 12345, size = 60, titel = 'Unknown Data')






















#------------------------------------------------------
#                 Wavelet transform driver
#------------------------------------------------------
#Wavelet transform data
def waveize(data, wavelet = 'Haar'):
    """
    Function for transforming the data into wavelet transformed dataframe.

    SYNTAX:
        waveize(data, unik):

    INPUT:
        data: pandas dataframe

        unik: list or numpy array listing the unique headers in the dataframe to wavelet transform.
            List of column headers

    RETURN:
        df: wavelet transformed dataframe

    ex:
        wavelet data and estimate classes.
        >>> dafr = waveize(data, unik[:100])
        >>> [sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(dafr, 20, rs=12345, titel ='Wavelets')
    
    """
    '''
    #df = pd.DataFrame(index=data[data.instnr==unik[0]].dato)
    df = pd.DataFrame(index=data.columns)
    for i in unik:
        a = data[data.columns==i]
        signal = wave(a, wavelet)
        df[str(i)] = signal

    return df
    '''

    wt = pd.DataFrame()#index = data.index)

    for i in data.columns:
        tmp = []
        [slet1, slet2, keep] = wavelets.wave(data[i], wavelet)
        for j in range(len(keep)):
            tmp = tmp+list(keep[j])
	#len(tmp)
        wt[i] = tmp

    return wt

    #wav =waveize(mc, wavelet = 'Coif16')
    #cv(, 21, titel = 'MC Wavelets Transformed Data')

    

'''
#Scaling grapher
[mc, norm, std, mdiv] = scale(ava)
cv(std, 21, titel = 'Standardized Data')
cv(mdiv, 21, titel = 'Mean-Divided Data')
cv(mc, 21, titel = 'Mean-Centered Data')
cv(norm, 21, titel = 'Normalized Data')

'''

"Normal clustering plot."
def plot_clas(mean, model, data):
    #plotting normal classification 4 classes with overlay and legend.

    #[mean, model] = clustering.classify(norm, 4, threshold=2, rs=12345, plotting= 'No')

    plt.plot(mean[0], label = 'Cluster 0, obs: 18', color = 'k')
    plt.plot(mean[1], label = 'Cluster 1, obs:  4', color = 'g')
    plt.plot(mean[2], label = 'Cluster 2, obs: 12', color = 'r')
    plt.plot(mean[3], label = 'Cluster 3, obs: 15', color = 'b')

    titel = 'Normalized Clustering. RS: 12345'#+str(rs)
    plt.title(titel, fontsize = 60)
    plt.tick_params(axis='both', which='major', labelsize=str(60*(2/5)))
    plt.xlabel('Date', fontsize=60*(5/6))
    #plt.ylabel('Value', fontsize=60*(5/6))
    #plt.ylim(0,1)

    plt.grid()
    plt.legend(loc='upper right', fontsize=60*(2/5))
    plt.show(block=False)

    plt.figure()
  
    #col = ['k','g','r','b']
    #for i in range(4):
    #    plt.plot(norm[norm.columns[model.labels_==i]], color = col[i], linewidth = .5)
    plt.plot(mean[0], label = 'Cluster 0, obs: 18', color = 'k', linewidth = 2)
    plt.plot(mean[1], label = 'Cluster 1, obs:  4', color = 'g', linewidth = 2)
    plt.plot(mean[2], label = 'Cluster 2, obs: 12', color = 'r', linewidth = 2)
    plt.plot(mean[3], label = 'Cluster 3, obs: 15', color = 'b', linewidth = 2)
    
    plt.plot(data[data.columns[model.labels_==0]], label = '_nolegend_', color = 'k', linewidth = .25)
    plt.plot(data[data.columns[model.labels_==1]], label = '_nolegend_', color = 'g', linewidth = .25)
    plt.plot(data[data.columns[model.labels_==2]], label = '_nolegend_', color = 'r', linewidth = .25)
    plt.plot(data[data.columns[model.labels_==3]], label = '_nolegend_', color = 'b', linewidth = .25)
        
    plt.grid()
    plt.legend(loc='upper right', fontsize=60*(2/5))

    titel = 'Normalized Clustering Overlay Fit. RS: 12345'#+str(rs)
    plt.title(titel, fontsize = 60)
    plt.tick_params(axis='both', which='major', labelsize=str(60*(2/5)))
    plt.xlabel('Date', fontsize=60*(5/6))
    #plt.ylabel('Value', fontsize=60*(5/6))
    #plt.ylim(0,1)
    plt.show(block=False)


"Back to original plot"

def plot_org(data, model):
    size = 30

   
    plt.figure()
    plt.plot(data[data.columns[model.labels_==0]], color = 'k', linewidth=.45, label='_nolegend_') #, label='Cluster 0, obs: 18'
    plt.plot(data[data.columns[model.labels_==1]], color = 'g', linewidth=1., label='_nolegend_') # , label='Cluster 1, obs: 4'
    plt.plot(data[data.columns[model.labels_==2]], color = 'r', linewidth=.5, label='_nolegend_') # , label='Cluster 2, obs: 12'
    plt.plot(data[data.columns[model.labels_==3]], color = 'b', linewidth=.5, label='_nolegend_') # , label='Cluster 3, obs: 15'

    plt.plot(data['102'], color = 'k', label='Cluster 0, obs: 18', linewidth=.45)
    plt.plot(data['114'], color = 'g', label='Cluster 1, obs: 4', linewidth=1.)
    plt.plot(data['101'], color = 'r', label='Cluster 2, obs: 12', linewidth=.5)
    plt.plot(data['111R'], color = 'b', label='Cluster 3, obs: 15', linewidth=.5)

    plt.tick_params(axis='both', which='major', labelsize=str(size*(2/5)))
    plt.title('Cluster Mapping to Original Data', fontsize=size)
    plt.xlabel('Date', fontsize = (5/6)*size)
    plt.ylabel('Mw', fontsize = (5/6)*size)
    plt.grid()
    plt.legend(loc='upper right', fontsize=size*(2/3))
    plt.show(block=False)
    



#All wavelet plots for the classification
def wav_plot(mc, mdiv, std, norm, ava):
    unik = ava.columns
    wav =waveize(mc, unik, wavelet = 'Coif16')
    cv(wav, 21, titel = 'Mean-Centered Wavelet Transformed Data')
    wav =waveize(mdiv, unik, wavelet = 'Coif16')
    cv(wav, 21, titel = 'Mean-Divided Wavelet Transformed Data')
    wav =waveize(std, unik, wavelet = 'Coif16')
    cv(wav, 21, titel = 'Standardized Wavelet Transformed Data')
    wav =waveize(norm, unik, wavelet = 'Coif16')
    cv(wav, 21, titel = 'Normalized Wavelet Transformed Data')
    wav =waveize(ava, unik, wavelet = 'Coif16')
    cv(wav, 21, titel = 'Original Wavelet Transformed Data')




#cluster balance table
def balance():
    print('---* Mean-Centered *---')
    print('---* * *---')
    print('---* * *---')
    wav =waveize(mc, unik, wavelet = 'Coif16')
    [mean, model] = clustering.classify(wav, 4, rs=12345, plotting= 'No')
    auxil.cls_dst(model)
    print()
    print('---* Mean-Divided 4 *---')
    print('---* * *---')
    print('---* * *---')
    wav =waveize(mdiv, unik, wavelet = 'Coif16')
    [mean, model] = clustering.classify(wav, 4, rs=12345, plotting= 'No')
    auxil.cls_dst(model)
    print()
    print('---* Mean-Divided 5 *---')
    print('---* * *---')
    print('---* * *---')
    wav =waveize(mdiv, unik, wavelet = 'Coif16')
    [mean, model] = clustering.classify(wav, 5, rs=12345, plotting= 'No')
    auxil.cls_dst(model)
    print()
    print('---* Standardized 4 *---')
    print('---* * *---')
    print('---* * *---')
    wav =waveize(std, unik, wavelet = 'Coif16')
    [mean, model] = clustering.classify(wav, 4, rs=12345, plotting= 'No')
    auxil.cls_dst(model)
    print()
    print('---* Normalized 4 *---')
    print('---* * *---')
    print('---* * *---')
    wav =waveize(norm, unik, wavelet = 'Coif16')
    [mean, model] = clustering.classify(wav, 4, rs=12345, plotting= 'No')
    auxil.cls_dst(model)
    print()
    print('---* Original *---')
    print('---* * *---')
    print('---* * *---')
    wav =waveize(ava, unik, wavelet = 'Coif16')
    [mean, model] = clustering.classify(wav, 4, rs=12345, plotting= 'No')
    auxil.cls_dst(model)


    #Wavelet clusters
    size = 30
    plt.figure()
    plt.plot(mean[0].values, color = 'k')
    plt.plot(mean[1].values, color = 'g')
    plt.plot(mean[2].values, color = 'r')
    plt.plot(mean[3].values, color = 'b')

    plt.tick_params(axis='both', which='major', labelsize=str(size*(2/5)))
    plt.title('Wavelet Clustering', fontsize=size)
    plt.xlabel('Date', fontsize = (5/6)*size)
    plt.ylabel('Mw', fontsize = (5/6)*size)
    plt.grid()
    plt.legend(loc='upper right', fontsize=size*(2/3))
    plt.show(block=False)

    plt.plot(data[data.columns[model.labels_==0]], color = 'k', linewidth=.45, label='_nolegend_') #, label='Cluster 0, obs: 18'



def stop():
    print('... Engaging cleaning lady')
    se = None
    [mc, norm, std, mdiv] = scale(se_rdy)
    #se_rdy = None
    mc = None
    mdiv = None
    std = None
    se_rdy = None



    print('Missing data in NORM:', norm.isnull().values.any())

    print('Remove Nans')
    #remove nans from NORM data
    nans=norm.isnull().any()
    #only keep no-null
    nans = nans[nans==False]
    nani = nans.index
    norm = norm[nani]

    print('Missing data in NORM:', norm.isnull().values.any())



#virker kun til SE data!
def pca_data_prep(data):
    #scale data
    scaler = StandardScaler()
    scaler.fit(data)
    se_scale = scaler.transform(data)

    #pca on columns
    pca = PCA(0.95)
    pca.fit(se_scale)
    se_red = pca.fit_transform(se_scale)

    #columns for dataframe
    clm = np.arange(1,111,1)
    #dataframe
    pca_data = pd.DataFrame(se_red, columns = clm)

    return pca_data




def nfold(lst, splits):

    '''
    nfold is a method for splitting a list into n splits for cross-validation.
    it takes as input a list randomizes the list and removes n equally sized sublists from the list like n fold cross validation
    the output is dictionary with n different sub lists ready for input for clustering. 

    Syntax:

    final_list = nfold(lst, splits)
    
    input:
        lst = list to be split into sublists, columns (header) from Pandas DataFrame as list.
        lst = list(se.columns)

        splits = number of splits to perform

    output:
        final_list = dictionary with all sublists ready for clustering. 


    '''

    #initialize sets
    rlst =  np.random.permutation(lst)
    lst = set(lst)
    size = len(rlst)

    #initialize variables
    final_list = {}
    pcs = int(round(size/splits, 0))
    start = 0

    #create sublists using sets. created by moving a window across the list. 
    for i in range(splits):
        end = pcs+start
        drop = set(rlst[start:end])
        
        start = end
        final_list[i] = lst.difference(drop)
        #print(drop, 'Remainder',lst.difference(drop))
        #print(drop, 'Remainder',final_list[i])
    return final_list

    #cv_rdy = nfold(lst, 10)



def cv_nfold(data, cv_rdy, max_clusters=36, size = 30, titel = 'Normalized Data, 10 Fold Pseudo Cross-Validation'):
    #size = 30
    '''
    Create cv_rdy dictionary first using nfold

    Method for creating nfold cross validation of clustering problems using pseudo cross validation built on 4 indices
    MIA, CDI, DBI & Silhouette

    '''

    silhouette = pd.DataFrame()
    mia_ind = pd.DataFrame()
    cluster_db = pd.DataFrame()
    davies = pd.DataFrame()


    for i in cv_rdy:
        print('----------',str(i+1),'----------')
        [sil, mia, cdi, dbi, inerti, models, c_pct] = clusters(data[list(cv_rdy[i])], max_clusters, 12345)

        silhouette[i] = sil.values()
        mia_ind[i] = mia.values()
        cluster_db[i] = cdi.values()
        davies[i] = dbi.values()
    
     
    x = list(range(2,max_clusters+1,1))

    sil_pm = pd.DataFrame()
    mia_pm = pd.DataFrame()
    cdi_pm = pd.DataFrame()
    dbi_pm = pd.DataFrame()
         
    sil_pm['Upper'] = silhouette.max(axis=1)
    sil_pm['Lower'] = silhouette.min(axis=1)
    sil_pm['Mean'] = silhouette.mean(axis=1)

    mia_pm['Upper'] = mia_ind.max(axis=1)
    mia_pm['Lower'] = mia_ind.min(axis=1)
    mia_pm['Mean'] = mia_ind.mean(axis=1)

    cdi_pm['Upper'] = cluster_db.max(axis=1)
    cdi_pm['Lower'] = cluster_db.min(axis=1)
    cdi_pm['Mean'] = cluster_db.mean(axis=1)

    dbi_pm['Upper'] = davies.max(axis=1)
    dbi_pm['Lower'] = davies.min(axis=1)
    dbi_pm['Mean'] = davies.mean(axis=1)

    fig = plt.figure(figsize= (18.0, 10.0))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot(x, sil_pm['Mean'], 'b', linewidth=3, label = 'Silhouette Mean')
    ax.plot(x, dbi_pm['Mean'], 'r', linewidth=3, label = 'DBI Mean')
    ax.plot(x, mia_pm['Mean'], 'k', linewidth=3, label = 'MIA Mean')
    ax.plot(x, cdi_pm['Mean'], 'g', linewidth=3, label = 'CDI Mean')
             
    ax.plot(x, sil_pm['Lower'], 'b--', linewidth=1, label='_nolegend_')
    ax.plot(x, sil_pm['Upper'], 'b--', linewidth=1, label='_nolegend_')

    ax.plot(x, mia_pm['Lower'], 'k--', linewidth=1, label='_nolegend_')
    ax.plot(x, mia_pm['Upper'], 'k--', linewidth=1, label='_nolegend_')

    ax.plot(x, cdi_pm['Lower'], 'g--', linewidth=1, label='_nolegend_')
    ax.plot(x, cdi_pm['Upper'], 'g--', linewidth=1, label='_nolegend_')

    ax.plot(x, dbi_pm['Lower'], 'r--', linewidth=1, label='_nolegend_')
    ax.plot(x, dbi_pm['Upper'], 'r--', linewidth=1, label='_nolegend_')


    plt.title(titel, fontsize=size*(5/6))
    plt.xlabel('# of Clusters', fontsize=size*(4/6))
    plt.ylabel('Index Value', fontsize=size*(4/6))
    plt.legend(fontsize=size*(3/6), loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=size/2)
    plt.grid()

    plt.xticks(np.arange(2, max_clusters+1, 2))


    plt.show(block=False)

    #loc = 'C:/OneDrive/PhD/Writing/4 Cluster Stability/week plots/'+str(i)+'png'
    #plt.savefig(loc, dpi = 300)

    #return sil_pm, mia_pm, cdi_pm, dbi_pm

    #[sil_pm, mia_pm, cdi_pm, dbi_pm] = cv_nfold(norm, cv_rdy, max_clusters=20, size = 50)


#------------ Plot Backup cv_nfold ------------ Plot Backup cv_nfold ------------ Plot Backup cv_nfold ------------ Plot Backup cv_nfold ------------ Plot Backup cv_nfold 

#----------------------------------------------
#          Backup of plotting of cv-_nfold
#----------------------------------------------
def puha(sil_pm, dbi_pm, mia_pm, cdi_pm, size=60,  titel = 'Normalized Data, 10 Fold Pseudo Cross-Validation'):
    #titel = 'Normalized Data, 10 Fold Pseudo Cross-Validation'
    max_clusters = 36
    x = list(range(2,max_clusters+1,1))
    fig = plt.figure()
    #size = 30
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot(x, sil_pm['Mean'], 'b', linewidth=3, label = 'Silhouette Mean')
    ax.plot(x, dbi_pm['Mean'], 'r', linewidth=3, label = 'DBI Mean')
    ax.plot(x, mia_pm['Mean'], 'k', linewidth=3, label = 'MIA Mean')
    ax.plot(x, cdi_pm['Mean'], 'g', linewidth=3, label = 'CDI Mean')
             
    ax.plot(x, sil_pm['Lower'], 'b--', linewidth=1, label='_nolegend_')
    ax.plot(x, sil_pm['Upper'], 'b--', linewidth=1, label='_nolegend_')

    ax.plot(x, mia_pm['Lower'], 'k--', linewidth=1, label='_nolegend_')
    ax.plot(x, mia_pm['Upper'], 'k--', linewidth=1, label='_nolegend_')

    ax.plot(x, cdi_pm['Lower'], 'g--', linewidth=1, label='_nolegend_')
    ax.plot(x, cdi_pm['Upper'], 'g--', linewidth=1, label='_nolegend_')

    ax.plot(x, dbi_pm['Lower'], 'r--', linewidth=1, label='_nolegend_')
    ax.plot(x, dbi_pm['Upper'], 'r--', linewidth=1, label='_nolegend_')

    plt.legend(loc='upper right')

    plt.title(titel, fontsize=size*(5/6))
    plt.xlabel('# of Clusters', fontsize=size*(4/6))
    plt.ylabel('Index Value', fontsize=size*(4/6))
    plt.legend(fontsize=size*(3/6), loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=size/2)
    plt.grid()

    plt.xticks(np.arange(2, max_clusters+1, 2))
    
    plt.show(block=False)



def eig():
    #Maybe useless pca plot of eigenvalues.
    plt.subplot(111)
    xx = range(1,169)
    x80 = range(1,89)
    size = 48
    #Scale data
    scaler = StandardScaler()
    scaler.fit(se)
    se_scale = scaler.transform(se)

    #total pca
    pca_total = PCA()
    pca_total.fit(se_scale)
    plt.plot(pca_total.explained_variance_ratio_*100, label= '100% Retained Variance', linewidth = 5)

    #Retain 90% 
    pca_red= PCA(.90)
    pca_red.fit(se_scale)
    plt.plot(pca_red.explained_variance_ratio_*100, 'r', label= '90% Retained Variance', linewidth = 3)

    titel = 'PC Explained Varinace Contribution for Time'
    plt.title(titel, fontsize = size)
    plt.tick_params(axis='both', which='major', labelsize=str(size*(2/5)))
    plt.xlabel('Principal Component', fontsize=size*(5/6))
    plt.ylabel('Total Variance %', fontsize=size*(5/6))
    #plt.ylim(0,1)
        
    plt.grid()
    #plt.legend(loc='upper right', fontsize=size*(2/5))
    plt.legend(loc = 'top right', fontsize=size*(3/5))#, bbox_to_anchor=(1, 0.5), )

    #ax = plt.gca()
    #ax.set_ticks(8)
    
    #plt.xticks(xx)

    plt.show(block=False)

