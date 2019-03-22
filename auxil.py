# Code for computing, data, models and Graphs for Paper 3.
#
#
# @ Alexander Tureczek, Technical University of Denmark, 2017
#
# Description: Auxillery functions for cluster check

import numpy as np



#----------------------------------------------
#                Size of classes
#----------------------------------------------
def cls_dst(model):
    """ Function for displaying the size of each class.

        SYNTAX: cls_dst(model)

        model: is a model object from the sklearn.Cluster KMeans class.

        RETURN: None
    """
    #classes in model
    classes = model.labels_.view()
    #number of unique class labels
    unik = len(np.unique(classes))
    for i in range(unik):
            print('Class #',i, 'Size:', len(classes[classes==i]))

    '''
    for i in range(9):
	cls_dst(models[i+1])
	print('-----*',str(i+1),'*-----')
    '''
