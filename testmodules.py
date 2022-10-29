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


# In[input features data]
from dataset.dataset import *
suffix = '_Total_Italy_train'  
info_file = f"data{suffix}.csv" # change the filename
label_list = dataset(info_file)
datafile_name = f"saved_data{suffix}.pkl"
with open(datafile_name, 'rb') as f:
    data = pickle.load(f)

# In[feature module]
from extraction.features import *

X = feature(data, "m")  # second arguments only "s", "m", "sm"

y = label_list
Xtra = X
ytra = y

# In[split training dataset into 10 fold]
from sklearn.model_selection import train_test_split
X_trainset = []
y_trainset = []
X_testset = []
y_testset = []
for i in range(10):
    if i < 9: 
        a,b,c,d = train_test_split(X, y, train_size = 0.1*(i+1), random_state = 0, stratify = y)
        X_trainset.append(a)
        X_testset.append(b)
        y_trainset.append(c)
        y_testset.append(d)
    else:
        X_trainset.append(X)
        y_trainset.append(y)

# In[train model]
from models.oritrain import *

blr = [] # best logistic regression algorithm number
brf = [] # best random forest algorithm number
bgb = [] # best gradient boosting algorithm number
for i in range(10):
    a, b, c = oritrain(X_trainset[i], y_trainset[i], i, "m")
    blr.append(a)
    brf.append(b)
    bgb.append(c)
    print(f"Finished No{i} algorithms.")


# In[import test dataset]
from extraction.features import *
#from dataset.dataset import *
#from extraction.features import *
suffix = '_Liz_train' # change the filename _Total_Italy_valid , _Liz_train
info_file = f"data{suffix}.csv" 
label_list = dataset(info_file)

with open(f"saved_data{suffix}.pkl", 'rb') as f:
    data = pickle.load(f)

Xval = feature(data, "sm")  # second arguments only "s", "m", "sm"

yval = label_list

# In[validate the algorithm through test dataset]

from models.validate import *
#import test.validate
acc_values = []
logloss_values = []
sen_values = []
for i in range(10):
    acc, ll, sen = validate(Xval, yval, i, "sm", 3, 15, 39)  # change the feature category blr[i], brf[i], bgb[i]
    acc_values.append(acc)
    logloss_values.append(ll)
    sen_values.append(sen)
accdf = pd.DataFrame(acc_values)
lldf = pd.DataFrame(logloss_values)
sendf = pd.DataFrame(sen_values)
total = pd.concat([accdf, lldf, sendf]) # axis = 1
#total.to_csv("total_stamor.csv") # sta, mor, sta_mor

# In[make ROC curves for train and test dataset]
X_train = Xtra
y_train = ytra
X_test = Xval
y_test = yval
from plots.roc import *
fig = roc(X_train, y_train, X_test, y_test)
# fig.savefig(fname="test.svg",format="svg")

# In[output feature tables]
from features.fetable import *
newX = fetable(X_train, "sm") # "s", "m", "sm"
newX['label'] = y
#newX.to_csv(f"data{suffix}/total_features.csv", index=False)

# In[shapvalue]
import shap
import os
import matplotlib.pyplot as pl
from shapvalues.shaptable import *
X['label'] = y
shpdf, clf = shaptable(X) #output is the shap values by features, and the corresponding algorithm
shpdf.to_csv("shpdf1.csv") # output shap values table

#shap figure
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X.drop('label',1))
shap.summary_plot(shap_values[1], X.drop('label',1),max_display=15,show=False)
f=pl.gcf() # shap values figure
f.savefig("feature_imp.svg",format='svg') # save the figure into high quality or other format




















