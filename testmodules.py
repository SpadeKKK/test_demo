# test all modules

import os
import numpy as np
from scipy import stats
import itertools
import pandas as pd
import pymzml
import pickle
import matplotlib.pyplot as plt
import math
#os.chdir("/Users/duranbao/Desktop/Study_abroad/PhD_projects/ML_update/ML_project/")
os.chdir("C:\\Users\\hanyu\\Documents\\Duran\\Pre_AI\\20200313_new_anno\\ML_project") # identify the working directionary
suffix = '_Total_Italy_train'
info_file = f"data{suffix}.csv" # change the filename

from dataset.dataset import *
label_list = dataset(info_file)

# In[input features data]
datafile_name = f"saved_data{suffix}.pkl"
        
with open(datafile_name, 'rb') as f:
    data = pickle.load(f)

# In[feature module]
from extraction.features import *

X = feature(data, "sm")  # second arguments only "s", "m", "sm"

y = label_list
Xtra = X
ytra = y
# In[train model]
from models.oritrain import *
oritrain(X, y)

'''
blr = best_index_LR # best logistic regression number
brf = best_index_RF # best random forest number
bgb = best_index_GB # best gradient boosting number
'''

# In[import test dataset]

#from dataset.dataset import *
#from extraction.features import *
suffix = '_Total_Italy_valid' # change the filename
info_file = f"data{suffix}.csv" 
label_list = dataset(info_file)

with open(f"saved_data{suffix}.pkl", 'rb') as f:
    data = pickle.load(f)

Xval = feature(data, "sm")  # second arguments only "s", "m", "sm"

yval = label_list

# In[validate the algorithm through test dataset]

from models.validate import *
#import test.validate
acc = validate(Xval, yval)

# In[make ROC curves for train and test dataset]
X_train = Xtra
y_train = ytra
X_test = Xval
y_test = yval
from plots.roc import *
fig = roc(X_train, y_train, X_test, y_test)
fig.savefig(fname="test.svg",format="svg")

# In[output feature tables]
from features.fetable import *
newX = fetable(X_train, "sm") # "s", "m", "sm"
newX['label'] = y
newX.to_csv(f"data{suffix}/total_features.csv", index=False)

# In[shapvalue]




