# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:46:46 2020

@author: admin
"""

# In[part1]
import os
import numpy as np
from scipy import stats
import itertools
import pandas as pd
import pymzml
import pickle
import matplotlib.pyplot as plt
import math
os.chdir("/Users/duranbao/Desktop/Study_abroad/PhD_projects/ML_update/ML_project/")
#os.chdir("C:\\Users\\hanyu\\Documents\\Duran\\Pre_AI\\20200313_new_anno\\ML_project") # identify the working directionary


clincal_info_file = f"Total_Italy_train.csv" # change the filename

if clincal_info_file is not None:
    # 1. Read in patient and sample id from clinical info file.

    clinical_info = pd.read_csv(clincal_info_file)
    clinical_info_selected = clinical_info[(clinical_info['Current criteria']=='TB') | (clinical_info['Current criteria']=='Not TB')] 

    label_list = clinical_info_selected['Current criteria'].values
    print(f"Propotion of positive samples: {np.mean(label_list=='TB')}")

    label_list = clinical_info_selected['Current criteria'].values
    label_list = (label_list == 'TB').astype('uint8') # Convert label 'Confirmed TB' and 'Unlikely TB' to 1 and 0

    def format_sample_id(x):
        return str(x).replace(' ', '').replace(')', '').replace('(', '_')
    clinical_info_selected_sample_id = clinical_info_selected['Sample ID'].values
    sample_id_list = [x for x in clinical_info_selected_sample_id]

else:
    label_list = None
    file_list = os.listdir(mzml_path)
    sample_id_list = [os.path.splitext(file)[0] for file in file_list if os.path.splitext(file)[1].lower() == ".mzml"]



datafile_name = f"saved_data.pkl"
        
with open(datafile_name, 'rb') as f:
    data = pickle.load(f)


# In[import feature module]
from features.feature import *

X = feature(data)

y = label_list






