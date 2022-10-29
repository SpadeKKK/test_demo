#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:07:20 2022

@author: duranbao
"""

# input data internal and external
import os
import pandas as pd
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, roc_curve, f1_score
from statistics import mean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


os.chdir('/Users/duranbao/Desktop/duran temp/Study_abroad/PhD_projects/1_ML_update/old_new_features/')

## Italy dataset
Italy_total = pd.read_csv("Italy_total.csv",index_col=0).dropna()
Italy_label = Italy_total['label']

Italy = Italy_newdict['New_sta_mor_relative_features']
Italy_l = Italy_newdict['label']

# Split Training and testing datasets
X_training, test_X, y_training, test_y = train_test_split(Italy, 
                            Italy_l, 
                            test_size=0.2, 
                            random_state=0)
# model setting
clf = RandomForestClassifier(n_jobs = -1, max_features='auto',
                         n_estimators=500, max_depth=5, 
                         random_state=0)
model1 = clf.fit(X_training,y_training)
y_pred = model1.predict(test_X)
y_prob = model1.predict_proba(test_X)[:,1]
print(f"AUC: {roc_auc_score(test_y, y_prob)}") # test the code first, AUC: 0.9755


# external part: dynamic & test parts 

lizx_test = INDTS1['New_sta_mor_relative_features'].values[:, ]
lizy_test = INDTS1['label'].values[:, ]

cstbx_test = INDTS2['New_sta_mor_relative_features'].values[:, ]
cstby_test = INDTS2['label'].values[:, ]

drx_test = INDTS3['New_sta_mor_relative_features'].values[:, ]
dry_test = INDTS3['label'].values[:, ]


#  AL processing functions

# extract features corresponding to the labels
def outX(outy, train_X):
    out_X = train_X[train_X.index.isin(outy.index)].reindex(outy.index)
    return out_X
# sampling by even score distribution
def systematic_sampling(df, n): 
    step = len(df)/n
    indexes = np.round(np.arange(0, len(df), step=step))[0:n]
    systematic_sample = df.iloc[indexes]
    return systematic_sample

# balancing TRS by un-even score distribution sampling
def rs_bal(ini_X, ini_y, unlabel, label):
    clf1 = RandomForestClassifier()
    ini_X = pd.DataFrame(ini_X)
    ini_y = pd.DataFrame(ini_y)
    unlabel = pd.DataFrame(unlabel)
    label = pd.DataFrame(label)
    
    x_train = ini_X.append(unlabel)
    y_train =  ini_y.append(label) 
    mix = pd.concat([x_train,y_train], axis = 1)
    # balanced labels in sample pool
    pos = mix[mix.iloc[:,-1]==1]
    neg = mix[mix.iloc[:,-1]==0]
    
    if len(pos) > len(neg):
        pos = pos.sample(n = len(neg),random_state=0)
    else:
        neg = neg.sample(n = len(pos),random_state=0)
    
    final_y = pd.concat([pos,neg], axis=0)['label']
    final_X = outX(final_y,X_training)
    clf1 = clf1.fit(final_X, final_y)
    return final_X, final_y, clf1

# balancing TRS by even score distribution sampling
def al_bal(ini_X, ini_y, unlabel, label):
    #ini_X, ini_y, unlabel, label = ini_X, ini_y, dyn_X, dyn_y
    unlabel = pd.DataFrame(unlabel)
    label = pd.DataFrame(label)
    #label.colname = ['label']
    iniindex = [i for i in range(len(ini_y))]
    x_train = ini_X.set_axis(iniindex,axis = 'index')
    y_train = ini_y.set_axis(iniindex,axis = 'index')

    dynindex = [i for i in range(len(unlabel))]
    unlabel = unlabel.set_axis(dynindex,axis = 'index')
    label = label.set_axis(dynindex,axis = 'index')
    unlabel = unlabel.set_axis(x_train.columns, axis='columns')
    
    clf1 = RandomForestClassifier()
    clf1.fit(x_train, y_train)
    y_probab = clf1.predict_proba(unlabel)[:,1]
    p1 = 0.45 # range of uncertanity 0.47 to 0.53
    #p2 = 0.2
    uncrt_pt_ind = []
    for i in range(unlabel.shape[0]):
        if(y_probab[i] <= p1 or y_probab[i] >= 1-p1):
        #if((y_probab[i] <= p1 and y_probab[i] >= p2) or (y_probab[i] >= 1-p1 and y_probab[i] <= 1-p2)):
            uncrt_pt_ind.append(i)
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_train = x_train.append(unlabel.iloc[uncrt_pt_ind, :],ignore_index=True)
    y_train = y_train.append(label.iloc[uncrt_pt_ind,:],ignore_index=True) 
    '''
    unlabel = unlabel[~unlabel.index.isin(uncrt_pt_ind)]
    label = label[~label.index.isin(uncrt_pt_ind)]
    '''
    mix = pd.concat([x_train,y_train], axis = 1)
    # balanced labels in sample pool
    pos = mix[mix.iloc[:,-1]==1]
    neg = mix[mix.iloc[:,-1]==0]
    
    if len(pos) > len(neg):
        pos = systematic_sampling(pos,len(neg))
    else:
        neg = systematic_sampling(neg,len(pos))
    
    final_X = pd.concat([pos,neg], axis=0).iloc[:,:-1].values[:,]
    final_y = pd.concat([pos,neg], axis=0).iloc[:,-1].values[:,]
    return final_X, final_y, clf1




# calculate performance
def vali(clf, train_X, train_y, test_X, test_y):
    lst = []
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    y_prob = clf.predict_proba(test_X)[:,1]
    ac = clf.score(test_X, test_y)
    sen = recall_score(test_y, y_pred, pos_label=1)
    spe = recall_score(test_y, y_pred, pos_label=0)
    auc = roc_auc_score(test_y, y_prob)
    f1 = f1_score(test_y, y_pred)
    lst = [ac, sen, spe, auc, f1]
    return lst

# extracting the unbalancing dynamic data with fixed positive label ratio

def dynset(i, train_X, train_y):
    dyn_y_pos = train_y[train_y==1].sample(n = i,random_state=0)
    dyn_y_neg = train_y[train_y==0].sample(n = 100-i,random_state=0)
    dyn_y = dyn_y_pos.append(dyn_y_neg)
    dyn_X = outX(dyn_y, train_X)
    return dyn_X, dyn_y

train_X, test_X, train_y, test_y = train_test_split(Italy,Italy_l,train_size=0.8,random_state=0)
ini_X, dyn_X, ini_y, dyn_y = train_test_split(train_X, train_y, train_size=0.1,random_state=0,stratify=train_y)
ini_y_pos = ini_y[ini_y==1].sample(n = 5,random_state=0)
ini_y_neg = ini_y[ini_y==0].sample(n = 5,random_state=0)
ini_y = ini_y_pos.append(ini_y_neg)
ini_X = outX(ini_y, train_X)
dyn_X, dyn_y = dynset(5, train_X, train_y)

finalal_X, finalal_y, clf_al = al_bal(ini_X, ini_y, dyn_X, dyn_y)
finalrs_X, finalrs_y, clf_rs = rs_bal(ini_X, ini_y, dyn_X, dyn_y)

vali(clf, ini_X.append(dyn_X) , ini_y.append(dyn_y), test_X, test_y) # PL method
vali(clf, ini_X.append(finalrs_X) , ini_y.append(finalrs_y), test_X, test_y) # RS method
vali(clf, ini_X.append(finalal_X) , ini_y.append(pd.DataFrame(finalal_y)), test_X, test_y) # AL method

pd.DataFrame(finalal_y)

vali(clf, ini_X.append(dyn_X) , ini_y.append(dyn_y), drx_test, dry_test) # PL method
vali(clf, ini_X.append(final_X) , ini_y.append(final_y), drx_test, dry_test) # AL method







