# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 06:43:38 2020

@author: admin
"""

# In[1.1 Change working directory and data folder]

import os
import numpy as np
from scipy import stats
import itertools
import pandas as pd
import pymzml
import pickle
import matplotlib.pyplot as plt
import math
os.chdir("C:\\Users\\hanyu\\Documents\\Duran\\Pre_AI\\20200313_new_anno") # identify the working directionary
suffix = '_liz_train' # change the folder suffix

'''
suffix:
    _Italy_train
    _Italy_valid
    _CSTB
    _liz_train
    
file name:
    Total_Italy_train
    Total_Italy_valid
    CSTB
    liz_train
'''

mzml_path = f"data{suffix}/mzml_data_conversion_SRMspectra/" # folder path to mzml mass spec files
clincal_info_file = f"data{suffix}/liz_train.csv" # change the filename

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

# In[1.2 Transform mzml files into readable files]
# Get the list of mzML files

file_list = os.listdir(mzml_path)
sample_id_list = [os.path.splitext(file)[0] for file in file_list if os.path.splitext(file)[1].lower() == ".mzml"]

# 3. Extract features from mass spec data. List of SRM M/z: '622', '693', '822', '950', '1021', '1134', '1235'.
# We look for scantime corresponding to the maximum total ion concentration (TIC) of internal standard.
# Then, we extract ion intensity, relative abundance and ratio between target/internal standard of different M/z(s)
# Sample 314622_1.mzml might be mistyped as 314662_1.mzml. Please manually correct the file name before running the below code.

scan_range = 20 # Range around standard_max_index and tartget_max_index to extract feature
min_scan_minute = 10 # We are only interested in range (min_scan_minute, max_scan_minute)
max_scan_minute = 25 # We are only interested in range (min_scan_minute, max_scan_minute)
period_second = 60 # Windows to extract features for non-targeted region
nb_periods = int(np.ceil((max_scan_minute - min_scan_minute)*60/period_second))


maxstd_scanid_list = [] # list to store scan id corresponding to the maximum TIC of internal standard for each sample
maxstd_scantime_list = [] # list to store scantime corresponding to the maximum TIC of internal standard for each sample

# Next Features are new from TargetQCMS package in R
features_tic_peakmax_list = []
features_tic_peakarea_list = []
features_tic_peakratio_list = []
features_tic_peakshift_list = []
features_tic_elutionShift_list =  []
features_tic_similarity_list =  []
features_tic_symmetry_list = []
features_tic_jaggedness_list = []
features_tic_FWHM_list = []
features_tic_modality_list = []

features_transition_peakmax_list = []
features_transition_peakarea_list = []
features_transition_peakratio_list = []
features_transition_peakshift_list = []
features_transition_elutionShift_list =  []
features_transition_similarity_list =  []
features_transition_symmetry_list = []
features_transition_jaggedness_list = []
features_transition_FWHM_list = []
features_transition_modality_list = []

for sample_id in sample_id_list:
#    sample_id = 'ABWAE' # positive-PAVLE negative-MASIM 
#    sample_id = "136125" #test intensities contain 0
    mzml_file = os.path.join(mzml_path, sample_id + '.mzML')
    if os.path.exists(mzml_file):
        print(f"Processing {sample_id} ...")
        
        # Extract SRM
        run = pymzml.run.Reader(mzml_file)
        standard_scanid = []
        standard_scantime = []
        standard_intensity = []
        standard_tic = []
        target_scanid = []
        target_intensity = []
        target_tic = []
        for spectrum in run:
            if spectrum.ms_level == 2:
                selected_precursor = spectrum.selected_precursors[0]['mz']
                if np.round(selected_precursor, 2) == 802.37: # Precursor corresponding to Internal Standard
                    standard_scanid.append(spectrum.ID)
                    standard_scantime.append(float(spectrum.get_element_by_path(['scanList', 'scan', 'cvParam'])[0].get('value')))
                    standard_intensity.append(spectrum.i)
                    standard_tic.append(spectrum.TIC)
                if np.round(selected_precursor, 2) == 797.38: # Precursor corresponding to Target
                    target_scanid.append(spectrum.ID)
                    target_intensity.append(spectrum.i)
                    target_tic.append(spectrum.TIC)
    
        standard_tic = np.array(standard_tic)
        target_tic = np.array(target_tic)
        standard_scantime = np.array(standard_scantime)
        standard_max_index = np.argmax(standard_tic[standard_scantime < 20], axis=0) # index corresponding to max TIC of internal standard
        target_max_index = np.argmax(target_tic, axis=0) # index corresponding to max TIC of target
      
       
        ## Define the peak boundaries by own calculation, we assume that the standard peak obey normal distribution.
        ## The peak boundaries are the 95% confidence interval's lower and upper thresholds.
     
    
        temp = target_tic[target_max_index - 20:target_max_index + 20]
        temp1 = standard_tic[standard_max_index - 20:standard_max_index + 20]
        temp1 = (np.round(temp1,0)).astype(int)
        new = []
        for i in range(40):
            if i == 1:
                new = [1]*temp1[1]
            else:
                new.extend([i]*temp1[i])
        thr = math.floor(np.std(new)*1.96) # if mean scan is 0, the boundaries are (-thr, thr)
    
        ## define normalize function for next use
        def normalize(lst):
            if max(lst) != 0:
                return [(float(i)-min(lst))/(max(lst)-min(lst)) for i in lst]
            else:
                return lst
    
        ## 1.1 Target Maximum in peak boundaries
        features_tic_peakmax = []
        tmax = max(target_tic[standard_max_index - thr:standard_max_index + thr])
        features_tic_peakmax.append(tmax)
    
        
        ### 1.2 Target transitions Maximum in peak boundaries
        
        features_transition_peakmax = []
        temp1 = pd.DataFrame(target_intensity[standard_max_index-thr:standard_max_index+thr])
        temp = pd.DataFrame(standard_intensity[standard_max_index-thr:standard_max_index+thr])
        for x in range(7):
            features_transition_peakmax.append(max(temp1[x]))   
    
      
        ## 2.1 Peak Area between peak boundaries, considered as the summation of each scan's intensity
        features_tic_peakarea = []
        features_tic_peakarea.append(sum(target_tic[standard_max_index - thr:standard_max_index + thr]))
        
        ### 2.2 Peak Area of transitions between peak boundaries
        features_transition_peakarea = []
        for x in range(7):
            features_transition_peakarea.append(sum(temp1[x]))
        
        ## 3.1 Peak ratio: Peak area under the target/Peak area under the standard
        features_tic_peakratio = []
        if max(standard_tic[standard_max_index - thr:standard_max_index + thr]) != 0:
            features_tic_peakratio.append((sum(target_tic[standard_max_index - thr:standard_max_index + thr]))/(sum(standard_tic[standard_max_index - thr:standard_max_index + thr])))
        else:
            features_tic_peakratio.append(0)
        
        ### 3.2 Peak ratio of transitions
        features_transition_peakratio = []
        for x in range(7):
            if sum(temp[x]) == 0:
                features_transition_peakratio.append(0)             
            elif max(standard_tic[standard_max_index - thr:standard_max_index + thr]) != 0:
                features_transition_peakratio.append(sum(temp1[x])/(sum(temp[x])))
            else:
                features_transition_peakratio.append(0)       
        
        
        ## 4.1 PeakShift: difference between standard and target peak's retention time (scan)
        features_tic_peakshift = []
        if len(temp) != 0:
            temp = target_tic[standard_max_index - thr:standard_max_index + thr]
            features_tic_peakshift.append(abs([n for n,i in enumerate(temp) if i == max(temp)][0] - (len(temp)/2)))
        else:
            features_tic_peakshift.append(999)
        
        ### 4.2 PeakShift of transitions 
        features_transition_peakshift = []
        for x in range(7):
            if len(temp) != 0:
                temp = target_intensity[standard_max_index - thr:standard_max_index + thr]
                features_transition_peakshift.append(abs([n for n,i in enumerate(temp[x]) if i == max(temp[x])][0] - (len(temp[x])/2)))
            else:
                features_transition_peakshift.append(999)           
      
        ## 5.1 ElutionShift features 
        features_tic_elutionShift = []
        target_max_index = standard_max_index-thr+np.argmax(target_tic[standard_max_index-thr:standard_max_index+thr])
        temp = normalize(target_tic[target_max_index - thr:target_max_index + thr])
        features_tic_elutionShift.append(abs(temp[0]-temp[-1]))
    
        
        ### 5.2 ElutionShift_each_transition
        features_transition_elutionShift = []  
        temp = pd.DataFrame(target_intensity[target_max_index - thr:target_max_index + thr])
        for x in range(7):
            num = abs(normalize(temp[x])[len(temp)-1]-normalize(temp[x])[0])
            features_transition_elutionShift.append(num)
        
    
        ## 6.1 PeakShapeSimilarity
        features_tic_similarity = []
        sig1 = pd.Series(normalize(target_tic[target_max_index - thr:target_max_index + thr]))
        sig2 = pd.Series(normalize(standard_tic[standard_max_index - thr:standard_max_index + thr]))
        features_tic_similarity.append(abs(sig1).corr(sig2))
        
        ### 6.2 PeakShapeSimilarity_each_transition
        features_transition_similarity = []
        lst = target_intensity[target_max_index - thr:target_max_index + thr]
        lst = [lst[x].tolist() for x in range(len(lst))]
        for i in range(7):
            sigtr1 = pd.Series(normalize(list(zip(*lst))[i]))
            if max(sigtr1) == 0:
                features_transition_similarity.append(0)
            else:
                features_transition_similarity.append(abs(sigtr1).corr(sig2))
      
        ## 7.1 PeakSymmetry
        features_tic_symmetry = []
        pleft = pd.Series(normalize(target_tic[target_max_index-thr: target_max_index-1]))
        pright = pd.Series(normalize(target_tic[target_max_index+1: target_max_index+thr][::-1]))
        features_tic_symmetry.append(abs(pleft.corr(pright)))
    
    
        ### 7.2 PeakSymmetry_each_transition
        features_transition_symmetry = []
        pleft = target_intensity[target_max_index-thr: target_max_index-1] # transition at peaktime left part
        pright = target_intensity[target_max_index+1: target_max_index+thr][::-1] # transition at peaktime right part
        pleft = [pleft[x].tolist() for x in range(len(pleft))]
        pright = [pright[x].tolist() for x in range(len(pright))]
        for i in range(7):
            pltr = pd.Series(normalize(list(zip(*pleft))[i])) # In this code, we should use other list name, 'pltr != pleft'
            prtr = pd.Series(normalize(list(zip(*pright))[i]))
            features_transition_symmetry.append(abs(pltr).corr(prtr))
        features_transition_symmetry = np.nan_to_num(features_transition_symmetry, nan=0) # nan = 0 means this feature at ultra bad situation 
        features_transition_symmetry = features_transition_symmetry.tolist()
       
        ## 8.1 Jaggedness
        features_tic_jaggedness = []
        sig1 = pd.Series(normalize(standard_tic[standard_max_index-thr:standard_max_index+thr]))
        sig2 = pd.Series(normalize(target_tic[standard_max_index-thr:standard_max_index+thr]))
        dif = abs(sig1-sig2)
        new_dif = []
        for x in dif:
            if x<= 0.05*max(dif):
                new_dif.append(0)
            else:
                new_dif.append(x)
        features_tic_jaggedness.append((sum(new_dif)-0.05)/(len(new_dif)-0.05))
    
        ### 8.2 Jaggendeniss_each_transition
        features_transition_jaggedness = []
        temp = pd.DataFrame(standard_intensity[standard_max_index-thr:standard_max_index+thr])
        temp1 = pd.DataFrame(target_intensity[standard_max_index-thr:standard_max_index+thr])
        temp = temp.apply(normalize,axis=0)
        temp1 = temp1.apply(normalize,axis=0)
        dif = abs(temp.subtract(temp1))
        for x in range(7):
            features_transition_jaggedness.append((sum(dif[x])-0.05)/(len(dif[x])-0.05))
       
        ## 9.1 FWHM (Full width and half maximum)
        features_tic_FWHM = []
        if max(sig2) == 0:
            features_tic_FWHM.append(0)
        else: 
            pleft = [n for n,i in enumerate(sig2) if i - (max(sig2)/2) > 0][0]
            pright = [n for n,i in enumerate(sig2) if i - (max(sig2)/2) > 0][-1]
            features_tic_FWHM.append(pright-pleft)
        ### 9.2 FWHM_each_transition (Full width and half maximum)
        temp1[0]
        features_transition_FWHM = []
        for x in range(7):
            if max(temp1[x]) == 0:
                features_transition_FWHM.append(0)
            else:        
                pleft = [n for n,i in enumerate(temp1[x]) if i - (max(temp1[x])/2) > 0][0]
                pright = [n for n,i in enumerate(temp1[x]) if i - (max(temp1[x])/2) > 0][-1]
                features_transition_FWHM.append(pright-pleft)            
            
        ## 10.1 Modality
        features_tic_modality = []
        
        dif = sig1-sig2
        new_dif = []
        rais = []
        fall = []
        for x in dif:
            if abs(x) <= 0.05*max(dif): 
                new_dif.append(0)
            else: 
                new_dif.append(x)
        if sum(new_dif) == 0:
            features_tic_modality_list.append(1)
        else:
            rais = [n for n,i in enumerate(new_dif) if i > 0][0]
            fall = [n for n,i in enumerate(new_dif) if i <= 0][-1]
            if rais > fall:
                features_tic_modality.append(1)
            else:
                features_tic_modality.append(max(new_dif[rais:fall]))
        
        ### 10.2 Modality_each_transition
        features_transition_modality = []
        dif = temp.subtract(temp1)
        for i in range(7):
            dif[i].apply(lambda x: 0 if abs(x) <= 0.05*max(dif[i]) else x)
        for x in range(7):
            if sum(dif[x]) != 0:
                rais = [n for n,i in enumerate(dif[x]) if i > 0][0]
                fall = [n for n,i in enumerate(dif[x]) if i <= 0][-1]
                if rais <= fall:
                    features_transition_modality.append(round(max(dif[x][rais:fall]),4)) 
                else:
                    features_transition_modality.append(1)   
            else:
                features_transition_modality.append(1)                           
 
        # New features
        # tic part
        features_tic_peakmax_list.append(features_tic_peakmax[0])
        features_tic_peakarea_list.append(features_tic_peakarea[0])
        features_tic_peakratio_list.append(features_tic_peakratio[0])
        features_tic_peakshift_list.append(features_tic_peakshift[0])
        features_tic_elutionShift_list.append(features_tic_elutionShift[0])
        features_tic_similarity_list.append(features_tic_similarity[0])
        features_tic_symmetry_list.append(features_tic_symmetry[0])
        features_tic_jaggedness_list.append(features_tic_jaggedness[0])
        features_tic_FWHM_list.append(features_tic_FWHM[0])
        features_tic_modality_list.append(features_tic_modality[0])
        
        # transition part
        
        features_transition_peakmax_list.append(features_transition_peakmax)
        features_transition_peakarea_list.append(features_transition_peakarea)
        features_transition_peakratio_list.append(features_transition_peakratio)
        features_transition_peakshift_list.append(features_transition_peakshift)
        features_transition_elutionShift_list.append(features_transition_elutionShift)
        features_transition_similarity_list.append(features_transition_similarity)
        features_transition_symmetry_list.append(features_transition_symmetry)
        features_transition_jaggedness_list.append(features_transition_jaggedness)
        features_transition_FWHM_list.append(features_transition_FWHM)
        features_transition_modality_list.append(features_transition_modality)
        
    else:
        print(f"{sample_id} does not exist.")

# In[1.3 Save the data into folder]
        

#### new features start
features_tic_peakmax_list = np.array(features_tic_peakmax_list)
features_tic_peakarea_list = np.array(features_tic_peakarea_list)
features_tic_peakratio_list = np.array(features_tic_peakratio_list)
features_tic_peakshift_list = np.array(features_tic_peakshift_list)
features_tic_elutionShift_list = np.array(features_tic_elutionShift_list)
features_tic_similarity_list = np.array(features_tic_similarity_list)
features_tic_symmetry_list = np.array(features_tic_symmetry_list)
features_tic_jaggedness_list = np.array(features_tic_jaggedness_list)
features_tic_FWHM_list = np.array(features_tic_FWHM_list)
features_tic_modality_list = np.array(features_tic_modality_list)

features_transition_peakmax_list = np.array(features_transition_peakmax_list)
features_transition_peakarea_list = np.array(features_transition_peakarea_list)
features_transition_peakratio_list = np.array(features_transition_peakratio_list)
features_transition_peakshift_list = np.array(features_transition_peakshift_list)
features_transition_elutionShift_list = np.array(features_transition_elutionShift_list)
features_transition_similarity_list = np.array(features_transition_similarity_list)
features_transition_symmetry_list = np.array(features_transition_symmetry_list)
features_transition_jaggedness_list = np.array(features_transition_jaggedness_list)
features_transition_FWHM_list = np.array(features_transition_FWHM_list)

features_transition_modality_list = np.array(features_transition_modality_list)


### new features over

data = {       
        "sample_id": sample_id_list,
        "features_tic_peakmax_list": features_tic_peakmax_list,
        "features_tic_peakarea_list": features_tic_peakarea_list,
        "features_tic_peakratio_list": features_tic_peakratio_list,
        "features_tic_peakshift_list": features_tic_peakshift_list,
        "features_tic_elutionShift_list":features_tic_elutionShift_list,
        "features_tic_similarity_list":features_tic_similarity_list,
        "features_tic_symmetry_list":features_tic_symmetry_list,
        "features_tic_jaggedness_list":features_tic_jaggedness_list,
        "features_tic_FWHM_list": features_tic_FWHM_list,
        "features_tic_modality_list":features_tic_modality_list,
        
        "features_transition_peakmax_list": features_transition_peakmax_list,
        "features_transition_peakarea_list": features_transition_peakarea_list,
        "features_transition_peakratio_list": features_transition_peakratio_list,
        "features_transition_peakshift_list": features_transition_peakshift_list,
        "features_transition_elutionShift_list":features_transition_elutionShift_list,
        "features_transition_similarity_list":features_transition_similarity_list,
        "features_transition_symmetry_list":features_transition_symmetry_list,
        "features_transition_jaggedness_list":features_transition_jaggedness_list,
        "features_transition_FWHM_list":features_transition_FWHM_list,
        "features_transition_modality_list":features_transition_modality_list,
        }

os.makedirs(f"data{suffix}", exist_ok=True)
datafile_name = f"saved_data.pkl"
with open(f"data{suffix}/{datafile_name}", "wb") as f:
    pickle.dump(data, f)


# In[2.1 Training Data]
    
datafile_name = f"data{suffix}/saved_data.pkl"
        
with open(datafile_name, 'rb') as f:
    data = pickle.load(f)

# Select only features of sample in sample_id_list
selected_sample_index = [list(data['sample_id']).index(i) for i in sample_id_list]
#assert(np.array_equal(data['sample_id'][selected_sample_index], sample_id_list))

# features of tic part
features_tic_peakmax_list = data['features_tic_peakmax_list'][selected_sample_index]
#features_tic_relmax_list = data['features_tic_relmax_list'][selected_sample_index]
features_tic_peakarea_list = data['features_tic_peakarea_list'][selected_sample_index]
features_tic_peakratio_list = data['features_tic_peakratio_list'][selected_sample_index]
features_tic_peakshift_list = data['features_tic_peakshift_list'][selected_sample_index]
features_tic_elutionShift_list = data['features_tic_elutionShift_list'][selected_sample_index]
features_tic_similarity_list = data['features_tic_similarity_list'][selected_sample_index]
features_tic_symmetry_list = data['features_tic_symmetry_list'][selected_sample_index]
features_tic_jaggedness_list = data['features_tic_jaggedness_list'][selected_sample_index]
features_tic_FWHM_list = data['features_tic_FWHM_list'][selected_sample_index]
features_tic_modality_list = data['features_tic_modality_list'][selected_sample_index]

# features of transition part
features_transition_peakmax_list = data['features_transition_peakmax_list'][selected_sample_index]
#features_transition_relmax_list = data['features_transition_relmax_list'][selected_sample_index]
features_transition_peakarea_list = data['features_transition_peakarea_list'][selected_sample_index]
features_transition_peakratio_list = data['features_transition_peakratio_list'][selected_sample_index]
features_transition_peakshift_list = data['features_transition_peakshift_list'][selected_sample_index]
features_transition_elutionShift_list = data['features_transition_elutionShift_list'][selected_sample_index]
features_transition_similarity_list = data['features_transition_similarity_list'][selected_sample_index]
features_transition_symmetry_list = data['features_transition_symmetry_list'][selected_sample_index]
features_transition_jaggedness_list = data['features_transition_jaggedness_list'][selected_sample_index]
features_transition_FWHM_list = data['features_transition_FWHM_list'][selected_sample_index]
features_transition_modality_list = data['features_transition_modality_list'][selected_sample_index]



intensity_list = [] # list to store intensity features of selected ions
rel_list = [] # list to store relative abundance features of selected ions
intensity_sum_list = [] # list to store peak area features of selected ions
rel_sum_list = [] # list to store relative abundance based on peak area of selected ions
max_tic_range_list = [] # list to store featured based on Max TIC of target winthin a range around standard_max_index
std_tic_list = []
tar_tic_list = []

## new features
tic_peakmax_list = []
tic_peakarea_list = []
tic_peakratio_list = []
tic_peakshift_list = []
tic_elutionShift_list = []
tic_similarity_list = []
tic_symmetry_list = []
tic_jaggedness_list = []
tic_FWHM_list = []
tic_modality_list = []

transition_peakmax_list = []
transition_peakarea_list = []
transition_peakratio_list = []
transition_peakshift_list = []
transition_elutionShift_list = []
transition_similarity_list = []
transition_symmetry_list = []
transition_jaggedness_list = []
transition_FWHM_list = []
transition_modality_list = []    
                           

## We can choose any transition features here.
for x in range(7):
    peakmax = features_transition_peakmax_list[:,x].reshape(-1,1)
    peakarea = features_transition_peakarea_list[:,x].reshape(-1,1)
    peakratio = features_transition_peakratio_list[:,x].reshape(-1,1)
    peakshift = features_transition_peakshift_list[:,x].reshape(-1,1)
    eshift = features_transition_elutionShift_list[:,x].reshape(-1,1)
    similarity = features_transition_similarity_list[:,x].reshape(-1,1)
    symmetry = features_transition_symmetry_list[:,x].reshape(-1,1)
    jaggedness = features_transition_jaggedness_list[:,x].reshape(-1,1)
    FWHM = features_transition_FWHM_list[:,x].reshape(-1,1)
    modality = features_transition_modality_list[:,x].reshape(-1,1)
  
    
    
    transition_peakmax_list.append(peakmax)
    transition_peakarea_list.append(peakarea)
    transition_peakratio_list.append(peakratio)
    transition_peakshift_list.append(peakshift)
    transition_elutionShift_list.append(eshift)
    transition_similarity_list.append(similarity)
    transition_symmetry_list.append(symmetry)
    transition_jaggedness_list.append(jaggedness)
    transition_FWHM_list.append(FWHM)
    transition_modality_list.append(modality)


tic_peakmax_list = features_tic_peakmax_list.reshape(-1,1)
#tic_relmax_list = features_tic_relmax_list.reshape(-1,1)
tic_peakarea_list = features_tic_peakarea_list.reshape(-1,1)
tic_peakratio_list = features_tic_peakratio_list.reshape(-1,1)
tic_peakshift_list = features_tic_peakshift_list.reshape(-1,1)
tic_elutionShift_list = np.reshape([round(x,8) for x in features_tic_elutionShift_list],(-1,1))
tic_similarity_list = features_tic_similarity_list.reshape(-1,1)
tic_symmetry_list = features_tic_symmetry_list.reshape(-1,1)
tic_jaggedness_list = features_tic_jaggedness_list.reshape(-1,1)
tic_FWHM_list = features_tic_FWHM_list.reshape(-1,1)
tic_modality_list = np.reshape(features_tic_modality_list,(-1,1))

transition_peakmax_list = np.concatenate(transition_peakmax_list, axis=1)
#transition_relmax_list = np.concatenate(transition_relmax_list, axis=1)
transition_peakarea_list = np.concatenate(transition_peakarea_list, axis=1)
transition_peakratio_list = np.concatenate(transition_peakratio_list, axis=1)
transition_peakshift_list = np.concatenate(transition_peakshift_list, axis=1)
transition_elutionShift_list = np.concatenate(transition_elutionShift_list, axis=1)
transition_similarity_list = np.concatenate(transition_similarity_list, axis=1)
transition_symmetry_list = np.concatenate(transition_symmetry_list, axis=1)
transition_jaggedness_list = np.concatenate(transition_jaggedness_list, axis=1)
transition_FWHM_list = np.concatenate(transition_FWHM_list, axis=1)
transition_modality_list = np.concatenate(transition_modality_list, axis=1)

## We could pick different combination here by changing the 0 and 1, but please notice there must
## be only one '1' in each group.

feature_sta = 0
feature_mor = 0
feature_ms = 1
   

X=[]

if feature_mor:

    X = np.concatenate((
                    transition_elutionShift_list,
                    transition_similarity_list,
                    transition_symmetry_list,
                    transition_jaggedness_list,
                    transition_modality_list,
                    transition_FWHM_list,
                    tic_elutionShift_list,
                    tic_similarity_list,
                    tic_symmetry_list,
                    tic_jaggedness_list,
                    tic_modality_list,                          
                    tic_FWHM_list), axis=1)
    print('morphological features finished')
if feature_sta:
    X = np.concatenate((
                    transition_peakmax_list,
                    transition_peakarea_list,
                    transition_peakshift_list,                            
                    transition_peakratio_list,
                    tic_peakratio_list,
                    tic_peakmax_list,
                    tic_peakarea_list,
                    tic_peakshift_list,  
), axis=1)
    print('statistical features finished')
if feature_ms:
    X = np.concatenate((
                    tic_peakratio_list,
                    tic_elutionShift_list,
                    tic_similarity_list,
                    tic_symmetry_list,
                    tic_jaggedness_list,
                    tic_modality_list, 
                    transition_peakratio_list,
                    transition_elutionShift_list,
                    transition_similarity_list,
                    transition_symmetry_list,
                    transition_jaggedness_list,
                    transition_modality_list,
                    tic_peakmax_list,
                    tic_peakarea_list,
                    tic_peakshift_list,                            
                    tic_FWHM_list,
                    transition_peakmax_list,
                    transition_peakarea_list,
                    transition_peakshift_list,                            
                    transition_FWHM_list), axis=1)
    print('both mor and sta finished')

X = np.nan_to_num(X, nan=-9999)

y = label_list
X_test3 = X
y_test3 = y
# In[output csv features table, notice to match the feature names]
a1 = [
"tic_peakratio",
"tic_elutionShift",
"tic_similarity",
"tic_symmetry",
"tic_jaggedness",
"tic_modality"]

a2 = [
"transition_peakratio_",
"transition_elutionShift_",
"transition_similarity_",
"transition_symmetry_",
"transition_jaggedness_",
"transition_modality_"]
a22 = []
for i in a2:
    for j in range(7):
        a22.append(i+str(j+1))

a3 =[
"tic_peakmax",
"tic_peakarea",
"tic_peakshift",                            
"tic_FWHM"]

a4 = [
"transition_peakmax_",
"transition_peakarea_",
"transition_peakshift_",                            
"transition_FWHM_"]
a42 = []
for i in a4:
    for j in range(7):
        a42.append(i+str(j+1))


a5 = a1 + a22 + a3+ a42

newX = pd.DataFrame(X, columns = a5)
newX['sample_id'] = sample_id_list
newX['label'] = y

newX.to_csv(f"data{suffix}/total_features.csv", index=False)

# In[2.2 base modelling and training model]
# Unless we want to optimize the parameters of models, or this step only need to run once. 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve

estimators = [] # list contains different models
parameters = {'penalty': ['l2'], 'C': [100.0, 10.0, 1.0]}
for penalty in parameters['penalty']:
    for C in parameters['C']:
        clf = LogisticRegression(n_jobs=-1, max_iter=500, penalty=penalty, C=C,
                                 random_state=0)
        estimators.append(("LR", clf))

parameters = {'n_estimators': [100, 500], 'max_depth': [2, 3, 5], 'max_features':[None, 'auto']}
for n_estimators in parameters['n_estimators']:
    for max_depth in parameters['max_depth']:
        for max_features in parameters['max_features']: 
            clf = RandomForestClassifier(n_jobs=-1, max_features=max_features,
                                         n_estimators=n_estimators, max_depth=max_depth,
                                         random_state=0)
            estimators.append(("RF", clf))
            
parameters = {'n_estimators': [100, 500], 'max_depth': [2, 3, 5], 'max_features': ['auto', None],
              'learning_rate': [0.01, 0.001]}
for n_estimators in parameters['n_estimators']:
    for max_depth in parameters['max_depth']:
        for max_features in parameters['max_features']: 
            for learning_rate in parameters['learning_rate']: 
                clf = GradientBoostingClassifier(random_state=0, subsample=1.0)
                estimators.append(("GB", clf))    

# Define dict to store metrics and predicted labels
accuracy_scores = {}
f_scores = {}
sensitivity_scores = {}
specificity_scores = {}
auc_roc = {}
y_prob_final = {}

def setup_scores(i):
    if i not in accuracy_scores:
        accuracy_scores[i] = []
    if i not in f_scores:
        f_scores[i] = []
    if i not in sensitivity_scores:
        sensitivity_scores[i] = []
    if i not in specificity_scores:
        specificity_scores[i] = []
    if i not in auc_roc:
        auc_roc[i] = []
    if i not in y_prob_final:
        y_prob_final[i] = np.ones(len(label_list))*-1

from sklearn.model_selection import StratifiedKFold
n_repeats =10
    
shuffle_split = StratifiedKFold(n_splits=n_repeats, shuffle=True, random_state=0)
fold = 0
for train_index, test_index in shuffle_split.split(X=X, y=y):
    fold += 1
    print(f"Running fold {fold} of {n_repeats}...")
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    i = 0
    # Performance of machine learning models
    for (model_name, clf) in estimators:
        i += 1
        clf.fit(X_train, y_train)
        y_prob_train = clf.predict_proba(X_train)[:, 1]
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)  
        setup_scores(i)
        accuracy_scores[i].append([accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)])
        f_scores[i].append([f1_score(y_train, y_pred_train), f1_score(y_test, y_pred_test)])
        sensitivity_scores[i].append([recall_score(y_train, y_pred_train, pos_label=1), recall_score(y_test, y_pred_test, pos_label=1)])
        specificity_scores[i].append([recall_score(y_train, y_pred_train, pos_label=0), recall_score(y_test, y_pred_test, pos_label=0)])
        auc_roc[i].append([roc_auc_score(y_train, y_prob_train), roc_auc_score(y_test, y_prob_test)])
        np.put(y_prob_final[i], test_index, y_prob_test)
    
    

# In[2.4 Save the models]

# Save base models for inference
# If we want to save model generate by different features, we could change the numbers in pathway.
# However we should remember the number's meaning or using true name instead.
save_model = True
if save_model:
    os.makedirs(f"model{suffix}3/sample", exist_ok=True) # change number 1: Mor; 2: Sta; 3: Mor&Sta features
    
    i = 0
    for (model_name, clf) in estimators:
        i += 1
        clf.fit(X, y) # Rerain the models on the whole dataset
        with open(f"model{suffix}3/sample/base_model_{i}.pkl", "wb") as f: # change number
            pickle.dump(clf, f)
    print("\nBase models saved.")

# In[3.1 load train best model to make prediction]
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve, log_loss

model_path = f"model_Italy_train3/sample/" # Remember to choose the correct model path
#model_name = "base_model_6"
model_names = ["base_model_3","base_model_15","base_model_39"]
threshold = 0.5
test_Dict = {"base_model_3":"LR", "base_model_15":"RF", "base_model_39":"GB"}
# LR = Logistic Regression; RF = Random Forest; GB = Gradient Boosting

for model_name in model_names:
    if "base" in model_name: # The model is base model
        with open(os.path.join(model_path, f"{model_name}.pkl"), 'rb') as f:
            clf = pickle.load(f)
            y_prob = clf.predict_proba(X)[:, 1]
            y_pred = (y_prob > threshold).astype('uint8')
            print('Accuracy in ', test_Dict[model_name], accuracy_score(y, y_pred))
            print('F1_score', test_Dict[model_name], f1_score(y, y_pred))
            print('Sensitivity', test_Dict[model_name], recall_score(y, y_pred, pos_label=1))
            print('Specificity', test_Dict[model_name], recall_score(y, y_pred, pos_label=0))
            print('AUC', test_Dict[model_name], roc_auc_score(y, y_prob))
            print("LOG_LOSS", test_Dict[model_name], log_loss(y, y_pred))


# In[make total performance table]
import os
import numpy as np
from scipy import stats
import itertools
import pandas as pd
import pymzml
import pickle
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve, log_loss

os.chdir("C:\\Users\\hanyu\\Documents\\Duran\\Pre_AI\\20200313_new_anno") # identify the working directionary

values = [["_Italy_train", "Total_Italy_train.csv","MLTR"], ["_Italy_valid", "Total_Italy_valid.csv","MLVA"], 
          ["_liz_train", "liz_train.csv","INDTS1"],["_CSTB", "CSTB.csv","INDTS2"]]
results_table = pd.DataFrame(columns = ["Dataset", "Base Model Number", "Features", "Scores", "Models", "values"])
best_index = {}
for suf, file_name, subset in values:
#    suf = "_Italy_train"
#    file_name = "Total_Italy_train.csv"
    
    suffix = suf # change the folder suffix
    print(suffix)
    model_save={}

    mzml_path = f"data{suffix}/mzml_data_conversion_SRMspectra/" # folder path to mzml mass spec files
    clincal_info_file = f"data{suffix}/{file_name}" # change the filename
    
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

    datafile_name = f"data{suffix}/saved_data.pkl"
            
    with open(datafile_name, 'rb') as f:
        data = pickle.load(f)
    
    # Select only features of sample in sample_id_list
    selected_sample_index = [list(data['sample_id']).index(i) for i in sample_id_list]
    #assert(np.array_equal(data['sample_id'][selected_sample_index], sample_id_list))
    
    # features of tic part
    features_tic_peakmax_list = data['features_tic_peakmax_list'][selected_sample_index]
    features_tic_peakarea_list = data['features_tic_peakarea_list'][selected_sample_index]
    features_tic_peakratio_list = data['features_tic_peakratio_list'][selected_sample_index]
    features_tic_peakshift_list = data['features_tic_peakshift_list'][selected_sample_index]
    features_tic_elutionShift_list = data['features_tic_elutionShift_list'][selected_sample_index]
    features_tic_similarity_list = data['features_tic_similarity_list'][selected_sample_index]
    features_tic_symmetry_list = data['features_tic_symmetry_list'][selected_sample_index]
    features_tic_jaggedness_list = data['features_tic_jaggedness_list'][selected_sample_index]
    features_tic_FWHM_list = data['features_tic_FWHM_list'][selected_sample_index]
    features_tic_modality_list = data['features_tic_modality_list'][selected_sample_index]
    
    # features of transition part
    features_transition_peakmax_list = data['features_transition_peakmax_list'][selected_sample_index]
    features_transition_peakarea_list = data['features_transition_peakarea_list'][selected_sample_index]
    features_transition_peakratio_list = data['features_transition_peakratio_list'][selected_sample_index]
    features_transition_peakshift_list = data['features_transition_peakshift_list'][selected_sample_index]
    features_transition_elutionShift_list = data['features_transition_elutionShift_list'][selected_sample_index]
    features_transition_similarity_list = data['features_transition_similarity_list'][selected_sample_index]
    features_transition_symmetry_list = data['features_transition_symmetry_list'][selected_sample_index]
    features_transition_jaggedness_list = data['features_transition_jaggedness_list'][selected_sample_index]
    features_transition_FWHM_list = data['features_transition_FWHM_list'][selected_sample_index]
    features_transition_modality_list = data['features_transition_modality_list'][selected_sample_index]
    
    
    
    intensity_list = [] # list to store intensity features of selected ions
    rel_list = [] # list to store relative abundance features of selected ions
    intensity_sum_list = [] # list to store peak area features of selected ions
    rel_sum_list = [] # list to store relative abundance based on peak area of selected ions
    max_tic_range_list = [] # list to store featured based on Max TIC of target winthin a range around standard_max_index
    std_tic_list = []
    tar_tic_list = []
    
    ## new features
    tic_peakmax_list = []
    tic_peakarea_list = []
    tic_peakratio_list = []
    tic_peakshift_list = []
    tic_elutionShift_list = []
    tic_similarity_list = []
    tic_symmetry_list = []
    tic_jaggedness_list = []
    tic_FWHM_list = []
    tic_modality_list = []
    
    transition_peakmax_list = []
    transition_peakarea_list = []
    transition_peakratio_list = []
    transition_peakshift_list = []
    transition_elutionShift_list = []
    transition_similarity_list = []
    transition_symmetry_list = []
    transition_jaggedness_list = []
    transition_FWHM_list = []
    transition_modality_list = []    
                               
    
    ## We can choose any transition features here.
    for x in range(7):
        peakmax = features_transition_peakmax_list[:,x].reshape(-1,1)
        peakarea = features_transition_peakarea_list[:,x].reshape(-1,1)
        peakratio = features_transition_peakratio_list[:,x].reshape(-1,1)
        peakshift = features_transition_peakshift_list[:,x].reshape(-1,1)
        eshift = features_transition_elutionShift_list[:,x].reshape(-1,1)
        similarity = features_transition_similarity_list[:,x].reshape(-1,1)
        symmetry = features_transition_symmetry_list[:,x].reshape(-1,1)
        jaggedness = features_transition_jaggedness_list[:,x].reshape(-1,1)
        FWHM = features_transition_FWHM_list[:,x].reshape(-1,1)
        modality = features_transition_modality_list[:,x].reshape(-1,1)
      
        
        
        transition_peakmax_list.append(peakmax)
        transition_peakarea_list.append(peakarea)
        transition_peakratio_list.append(peakratio)
        transition_peakshift_list.append(peakshift)
        transition_elutionShift_list.append(eshift)
        transition_similarity_list.append(similarity)
        transition_symmetry_list.append(symmetry)
        transition_jaggedness_list.append(jaggedness)
        transition_FWHM_list.append(FWHM)
        transition_modality_list.append(modality)
    
    
    tic_peakmax_list = features_tic_peakmax_list.reshape(-1,1)
    #tic_relmax_list = features_tic_relmax_list.reshape(-1,1)
    tic_peakarea_list = features_tic_peakarea_list.reshape(-1,1)
    tic_peakratio_list = features_tic_peakratio_list.reshape(-1,1)
    tic_peakshift_list = features_tic_peakshift_list.reshape(-1,1)
    tic_elutionShift_list = np.reshape([round(x,8) for x in features_tic_elutionShift_list],(-1,1))
    tic_similarity_list = features_tic_similarity_list.reshape(-1,1)
    tic_symmetry_list = features_tic_symmetry_list.reshape(-1,1)
    tic_jaggedness_list = features_tic_jaggedness_list.reshape(-1,1)
    tic_FWHM_list = features_tic_FWHM_list.reshape(-1,1)
    tic_modality_list = np.reshape(features_tic_modality_list,(-1,1))
    
    transition_peakmax_list = np.concatenate(transition_peakmax_list, axis=1)
    #transition_relmax_list = np.concatenate(transition_relmax_list, axis=1)
    transition_peakarea_list = np.concatenate(transition_peakarea_list, axis=1)
    transition_peakratio_list = np.concatenate(transition_peakratio_list, axis=1)
    transition_peakshift_list = np.concatenate(transition_peakshift_list, axis=1)
    transition_elutionShift_list = np.concatenate(transition_elutionShift_list, axis=1)
    transition_similarity_list = np.concatenate(transition_similarity_list, axis=1)
    transition_symmetry_list = np.concatenate(transition_symmetry_list, axis=1)
    transition_jaggedness_list = np.concatenate(transition_jaggedness_list, axis=1)
    transition_FWHM_list = np.concatenate(transition_FWHM_list, axis=1)
    transition_modality_list = np.concatenate(transition_modality_list, axis=1)

    values = [[1,0,0,1, "Statistical"], [0,1,0,2,"Morphological"], [0,0,1,3, "M+S"]]
    
    #this for loop goes through all the different combinations of feature_more, 
    #feature_sta, and feature_ms and saves the results in a dataframe (model_save)
    
    for feature1, feature2, feature3, count, feature_type in values:
        print(feature_type)
        feature_sta = feature1
        feature_mor = feature2
        feature_ms = feature3
        count = str(count)
        
        X=[]
        
        if feature_sta:
            X = np.concatenate((
                            transition_peakmax_list,
                            transition_peakarea_list,
                            transition_peakshift_list,                            
                            transition_peakratio_list,
                            tic_peakratio_list,
                            tic_peakmax_list,
                            tic_peakarea_list,
                            tic_peakshift_list), axis=1)
            print('statistical features finished')
        if feature_mor:
        
            X = np.concatenate((
                            transition_elutionShift_list,
                            transition_similarity_list,
                            transition_symmetry_list,
                            transition_jaggedness_list,
                            transition_modality_list,
                            transition_FWHM_list,
                            tic_elutionShift_list,
                            tic_similarity_list,
                            tic_symmetry_list,
                            tic_jaggedness_list,
                            tic_modality_list,         
                            tic_FWHM_list), axis=1)
            print('morphological features finished')
        if feature_ms:
            X = np.concatenate((
                            tic_peakratio_list,
                            tic_elutionShift_list,
                            tic_similarity_list,
                            tic_symmetry_list,
                            tic_jaggedness_list,
                            tic_modality_list, 
                            transition_peakratio_list,
                            transition_elutionShift_list,
                            transition_similarity_list,
                            transition_symmetry_list,
                            transition_jaggedness_list,
                            transition_modality_list,
                            tic_peakmax_list,
                            tic_peakarea_list,
                            tic_peakshift_list,                            
                            tic_FWHM_list,
                            transition_peakmax_list,
                            transition_peakarea_list,
                            transition_peakshift_list,                            
                            transition_FWHM_list), axis=1)
            print('both mor and sta finished')
        
        X = np.nan_to_num(X, nan=-9999)
        
        y = label_list
        
        z= feature_type
        model_save.update({count:[X,y,z]})

        if(suffix == "_Italy_train"):
            estimators = [] # list contains different models
            parameters = {'penalty': ['l2'], 'C': [100.0, 10.0, 1.0]}
            for penalty in parameters['penalty']:
                for C in parameters['C']:
                    clf = LogisticRegression(n_jobs=-1, max_iter=500, penalty=penalty, C=C,
                                             random_state=0)
                    estimators.append(("LR", clf))
            
            parameters = {'n_estimators': [100, 500], 'max_depth': [2, 3, 5], 'max_features':[None, 'auto']}
            for n_estimators in parameters['n_estimators']:
                for max_depth in parameters['max_depth']:
                    for max_features in parameters['max_features']: 
                        clf = RandomForestClassifier(n_jobs=-1, max_features=max_features,
                                                     n_estimators=n_estimators, max_depth=max_depth,
                                                     random_state=0)
                        estimators.append(("RF", clf))
                        
            parameters = {'n_estimators': [100, 500], 'max_depth': [2, 3, 5], 'max_features': ['auto', None],
                          'learning_rate': [0.01, 0.001]}
            for n_estimators in parameters['n_estimators']:
                for max_depth in parameters['max_depth']:
                    for max_features in parameters['max_features']: 
                        for learning_rate in parameters['learning_rate']: 
                            clf = GradientBoostingClassifier(random_state=0, subsample=1.0)
                            estimators.append(("GB", clf))    
            
            # Define dict to store metrics and predicted labels
            accuracy_scores = {}
            f_scores = {}
            sensitivity_scores = {}
            specificity_scores = {}
            auc_roc = {}
            y_prob_final = {}
            
            def setup_scores(i):
                if i not in accuracy_scores:
                    accuracy_scores[i] = []
                if i not in f_scores:
                    f_scores[i] = []
                if i not in sensitivity_scores:
                    sensitivity_scores[i] = []
                if i not in specificity_scores:
                    specificity_scores[i] = []
                if i not in auc_roc:
                    auc_roc[i] = []
                if i not in y_prob_final:
                    y_prob_final[i] = np.ones(len(label_list))*-1
            
            from sklearn.model_selection import StratifiedKFold
            n_repeats =10
                
            shuffle_split = StratifiedKFold(n_splits=n_repeats, shuffle=True, random_state=4)  # change the random seed to get SE
            fold = 0
            for train_index, test_index in shuffle_split.split(X=X, y=y):
                fold += 1
                print(f"Running fold {fold} of {n_repeats}...")
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]
                i = 0
                # Performance of machine learning models
                for (model_name, clf) in estimators:
                    i += 1
                    clf.fit(X_train, y_train)
                    y_prob_train = clf.predict_proba(X_train)[:, 1]
                    y_prob_test = clf.predict_proba(X_test)[:, 1]
                    y_pred_train = clf.predict(X_train)
                    y_pred_test = clf.predict(X_test)  
                    setup_scores(i)
                    accuracy_scores[i].append([accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)])
                    f_scores[i].append([f1_score(y_train, y_pred_train), f1_score(y_test, y_pred_test)])
                    sensitivity_scores[i].append([recall_score(y_train, y_pred_train, pos_label=1), recall_score(y_test, y_pred_test, pos_label=1)])
                    specificity_scores[i].append([recall_score(y_train, y_pred_train, pos_label=0), recall_score(y_test, y_pred_test, pos_label=0)])
                    auc_roc[i].append([roc_auc_score(y_train, y_prob_train), roc_auc_score(y_test, y_prob_test)])
                    np.put(y_prob_final[i], test_index, y_prob_test)
                
            
            # In [2.4 Save the models]
            
            # Save base models for inference
            # If we want to save model generate by different features, we could change the numbers in pathway.
            # However we should remember the number's meaning or using true name instead.
            save_model = True
            if save_model:
                os.makedirs(f"model{suffix}{count}/sample", exist_ok=True) # change number 1: Mor; 2: Sta; 3: Mor&Sta features
                
                i = 0
                for (model_name, clf) in estimators:
                    i += 1
                    clf.fit(X, y) # Rerain the models on the whole dataset
                    with open(f"model{suffix}{count}/sample/base_model_{i}.pkl", "wb") as f: # change number
                        pickle.dump(clf, f)
                print("\nBase models saved.")
                
                
            #Determine and save indices of best models    
                test_accuracy_scores = [np.mean(np.array(accuracy_scores[i])[:, 1]) for i in accuracy_scores.keys()]
                LR_loop = "LR" + count
                RF_loop = "RF" + count
                GB_loop = "GB" + count
                best_index_LR = str(np.argmax(test_accuracy_scores[0:2])+1)
                best_index_RF = str(np.argmax(test_accuracy_scores[3:14])+1)
                best_index_GB = str(np.argmax(test_accuracy_scores[15:38])+1)
                #This accounts for if all GB or all RF models have the same accuracy
                if(best_index_RF == "1"):
                    best_index_RF = "4" 
                if(best_index_GB == "1"):
                    best_index_GB = "16" 
                    
                        
            
                best_index.update({LR_loop:best_index_LR, RF_loop:best_index_RF, GB_loop:best_index_GB})



    for i in range(1,4): 
        print(i)
        model_path = f"model_Italy_train{i}/sample/" # Remember to choose the correct model path
        #model_name = "base_model_6"

        LR_loop = "LR" + str(i)
        RF_loop = "RF" + str(i)
        GB_loop = "GB" + str(i)

        model_names = [["base_model_" + str(best_index[LR_loop]), "LR"],["base_model_" + str(best_index[RF_loop]), "RF"],
                           ["base_model_"+ str(best_index[GB_loop]), "GB"]]
        threshold = 0.5
        test_Dict = {"base_model_" + str(best_index[LR_loop]):"LR", 
                         "base_model_"+ str(best_index[RF_loop]):"RF", "base_model_"+ str(best_index[GB_loop]):"GB"}
      

        for model_name, model_type in model_names:
            new_row = {}
            if "base" in model_name: # The model is base model
                with open(os.path.join(model_path, f"{model_name}.pkl"), 'rb') as f:
                    clf = pickle.load(f)
                    print(model_name)
                    X, y, z = model_save[str(i)]
                    
                    y_prob = clf.predict_proba(X)[:, 1]
                    y_pred = (y_prob > threshold).astype('uint8')
                    print('Accuracy in ', test_Dict[model_name], accuracy_score(y, y_pred))
                    results_table = results_table.append({"Dataset": subset, "Base Model Number": model_name, "Features": z, "Model": model_type, "Scores":"Accuracy",
                                            "values": accuracy_score(y, y_pred) }, ignore_index = True)
                    
    
                   
                    print('F1_score', test_Dict[model_name], f1_score(y, y_pred))
                    results_table = results_table.append({"Dataset": subset, "Features": z, 
                        "Model": model_type, "Base Model Number": model_name, "Scores":"F1","values": f1_score(y, y_pred)}, ignore_index = True)
                    
                    print('Sensitivity', test_Dict[model_name], recall_score(y, y_pred, pos_label=1))
                    
                    results_table = results_table.append({"Dataset": subset, "Features": z, 
                                    "Model": model_type, "Scores":"Sensitivity", "Base Model Number": model_name,
                                    "values": recall_score(y, y_pred, pos_label=1)}, ignore_index = True)
                                    
                    print('Specificity', test_Dict[model_name], recall_score(y, y_pred, pos_label=0))
                        
                    results_table = results_table.append({"Dataset": subset, "Features": z, 
                                        "Model": model_type, "Scores":"Specificity","Base Model Number": model_name,
                                        "values": recall_score(y, y_pred, pos_label=0)}, ignore_index = True)
                    
                    print('AUC', test_Dict[model_name], roc_auc_score(y, y_prob))
                    
                    results_table = results_table.append({"Dataset": subset, "Features": z, 
                                        "Model": model_type, "Scores":"AUC","Base Model Number": model_name,
                                        "values": roc_auc_score(y, y_prob)}, ignore_index = True)
                    
                    print("LOG_LOSS", test_Dict[model_name], log_loss(y, y_pred))
                    
                    results_table = results_table.append({"Dataset": subset, "Features": z, 
                                        "Model": model_type, "Scores":"LOG_LOSS","Base Model Number": model_name,
                                        "values": log_loss(y, y_pred)}, ignore_index = True)

results_table.to_csv('C:\\Users\\hanyu\\Documents\\Duran\\Pre_AI\\20200313_new_anno\\newtotal_perform6.csv', index = False)

# In[ROC curves]
# sta_features' model
X_train1 = X
y_train1 = y

# mor_features' model
X_train2 = X
y_train2 = y

# mor&sta_features' model
X_train3 = X
y_train3 = y

# Second, run the testing datasets X and y, which test1-3 is Italy_valid dataset and test4-6 is CSTB dataset
X_test1 = X
y_test1 = y

X_test2 = X
y_test2 = y

X_test3 = X
y_test3 = y

# In[ROC part2.]
#values = [["_Italy_train", "Total_Italy_train.csv"], ["_Italy_valid", "Total_Italy_valid.csv"], 
#          ["_CSTB", "CSTB.csv"]]
temp = [[1,1,11,16, "Statistical", "MATR"], [2,1,10,16, "Morphological", "MLVA"]]#, [1,2,8,16, "Mor&Sta", "INDTS2"]]
## generate three at the same time
for i, i1, i2, i3, feature, dataset in temp:
    X_train = f'X_train{i}'
    y_train = f'y_train{i}'
    
    X_test = f'X_test{i}'
    y_test = f'y_test{i}'
    model_path = f"model_Italy_train{i}/sample/" # Remember to choose the correct model path
    model_names = [f"base_model_{i1}",f"base_model_{i2}",f"base_model_{i3}"]
    threshold = 0.5
    classifiers = []
    for model_name in model_names:
        if "base" in model_name: # The model is base model
            with open(os.path.join(model_path, f"{model_name}.pkl"), 'rb') as f:
                clf = pickle.load(f)
                classifiers.append(clf)
                y_prob = clf.predict_proba(X_train)[:, 1]
                y_pred = (y_prob > threshold).astype('uint8')
    
            # Define a result table as a DataFrame
            result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
            
            # Train the models and record the results
            for cls in classifiers:
                model = cls.fit(X_train, y_train)
                yproba = model.predict_proba(X_test)[::,1]
                fpr, tpr, _ = roc_curve(y_test,  yproba)
                auc = roc_auc_score(y_test, yproba)
                result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                                    'fpr':fpr, 
                                                    'tpr':tpr, 
                                                    'auc':auc}, ignore_index=True)
            
            # Set name of the classifiers as index labels, don't forget change the title of ROC chart
            result_table.set_index('classifiers', inplace=True)
            
            fig = plt.figure(figsize=(8,8))
            
            for i in result_table.index:
                plt.plot(result_table.loc[i]['fpr'], 
                         result_table.loc[i]['tpr'], 
                         label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
                
            plt.plot([0,1], [0,1], color='orange', linestyle='--')
            
            plt.xticks(np.arange(0.0, 1.1, step=0.1))
            plt.xlabel("1-Specificity", fontsize=20)
            
            plt.yticks(np.arange(0.0, 1.1, step=0.1))
            plt.ylabel("Sensitivity", fontsize=20)
            
            plt.title(f'{feature}_features in \n {dataset} dataset', fontweight='bold', fontsize=30)
            plt.legend(prop={'size':13}, loc='lower right')
            fig.savefig(fname=f"test{i}.svg",format="svg") 
# In[test2]
X_train = X_train3
y_train = y_train3

X_test = X_test3
y_test = y_test3

#values = [["_Italy_train", "Total_Italy_train.csv"], ["_Italy_valid", "Total_Italy_valid.csv"], 
#          ["_CSTB", "CSTB.csv"]]
#temp = [[1,1,11,16, "Statistical", "MATR"], [2,1,10,16, "Morphological", "MLVA"], [1,2,8,16, "Mor&Sta", "INDTS1"]]
## generate three at the same time
#for i, i1, i2, i3, feature, dataset in temp:

model_path = f"model_Italy_train3/sample/" # Remember to choose the correct model path
model_names = [[f"base_model_2", "LR"],[f"base_model_8", "RF"],[f"base_model_16", "GB"]]
threshold = 0.5
classifiers = []
for model_name, name in model_names:
    if "base" in model_name: # The model is base model
        with open(os.path.join(model_path, f"{model_name}.pkl"), 'rb') as f:
            clf = pickle.load(f)
            classifiers.append(clf)
            y_prob = clf.predict_proba(X_train)[:, 1]
            y_pred = (y_prob > threshold).astype('uint8')

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
result_table['name'] = ['LR', 'RF', 'GB']
# Set name of the classifiers as index labels, don't forget change the title of ROC chart
result_table.set_index('name', inplace=True)

fig = plt.figure(figsize=(8,8))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("1-Specificity", fontsize=20)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("Sensitivity", fontsize=20)
#'Sta&Mor'
plt.title(f'Sta&Mor_features in \n INDTS1 dataset', fontweight='bold', fontsize=30)
plt.legend(prop={'size':25}, loc='lower right')
fig.savefig(fname=f"test9.svg",format="svg") # change the figures name here, the 'svg' format is easy to modified by illustrator software.


# In[generate ROC curves repeatably]

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

values = [["_Italy_train", "Total_Italy_train.csv"], ["_Italy_valid", "Total_Italy_valid.csv"], 
          ["_CSTB", "CSTB.csv"]]

classifiers = []
lst = ["Statistical", "Morphological", "M+S"]
model_save={}


for suf, file_name in values: 
    for index, feature_type in enumerate(lst):
        
        i = index+1
        model_path = f"model_Italy_train{i}/sample/" # Remember to choose the correct model path
        
        LR_loop =  "LR" + str(i)
        RF_loop =  "RF" + str(i)
        GB_loop =  "GB" + str(i)
        model_names = [["base_model_" + str(best_index[LR_loop]), "LR"],["base_model_" + str(best_index[RF_loop]), "RF"],
                           ["base_model_"+ str(best_index[GB_loop]), "GB"]]
        
        #UPDATE THIS LATER
        #model_names = [["base_model_1", "LR"],["base_model_4", "RF"],
        #                   ["base_model_20", "GB"]]
        threshold = 0.5
        
        #UPDATE THIS LATER
        test_Dict = {"base_model_" + str(best_index[LR_loop]):"LR", 
                         "base_model_"+ str(best_index[RF_loop]):"RF", "base_model_"+ str(best_index[GB_loop]):"GB"}
        
        
        #test_Dict = {"base_model_1":"LR", 
        #                 "base_model_4":"RF", "base_model_20":"GB"}
        # LR = Logistic Regression; RF = Random Forest; GB = Gradient Boosting
        

        for model_name, model_type in model_names:
            #loops through model types of M, S, M+S
            new_row = {}
            if "base" in model_name: # The model is base model
                with open(os.path.join(model_path, f"{model_name}.pkl"), 'rb') as f:
                    clf = pickle.load(f)
                    print(model_name)
                    X, y, z = model_save[suf + feature_type]
                    if(suf!="_Italy_train"):
                        classifiers.append([suf+feature_type+model_type ,clf,feature_type])

                    
                    
                    
                    model_save.update({(suffix + feature_type):[X,y,z]})
                    
                    
                    
                    y_prob = clf.predict_proba(X)[:, 1]
                    y_pred = (y_prob > threshold).astype('uint8')
                    
                    
                    
                    #Generates excel file of accuracy, sensitivity, F1, and specificity
                    print('Accuracy in ', test_Dict[model_name], accuracy_score(y, y_pred))
                    results_table = results_table.append({"Dataset": suf, "Base Model Number": model_name, "Features": z, "Model": model_type, "Scores":"Accuracy",
                                            "values": accuracy_score(y, y_pred) }, ignore_index = True)
                    
    
                   
                    print('F1_score', test_Dict[model_name], f1_score(y, y_pred))
                    results_table = results_table.append({"Dataset": suf, "Features": z, 
                        "Model": model_type, "Base Model Number": model_name, "Scores":"F1","values": f1_score(y, y_pred)}, ignore_index = True)
                    
                    print('Sensitivity', test_Dict[model_name], recall_score(y, y_pred, pos_label=1))
                    
                    results_table = results_table.append({"Dataset": suf, "Features": z, 
                                    "Model": model_type, "Scores":"Sensitivity", "Base Model Number": model_name,
                                    "values": recall_score(y, y_pred, pos_label=1)}, ignore_index = True)
                                    
                    print('Specificity', test_Dict[model_name], recall_score(y, y_pred, pos_label=0))
                        
                    results_table = results_table.append({"Dataset": suf, "Features": z, 
                                        "Model": model_type, "Scores":"Specificity","Base Model Number": model_name,
                                        "values": recall_score(y, y_pred, pos_label=0)}, ignore_index = True)
             
#results_table.to_csv('C:/Users/Michael/Desktop/Machine_Learning/Machine_Learning_Project_to_Michael - 7_1_20/Machine_Learning_Project_to_Michael/file.csv', index = False)                
                
#%%       

for key, cls, feature_type in classifiers:
     X_train, y_train, z_train = model_save["_Italy_train" + feature_type]
     X_test, y_test, z_test = model_save[key[:-2]]
                             
     model = cls.fit(X_train, y_train)
     yproba = model.predict_proba(X_test)[::,1]
     fpr, tpr, _ = roc_curve(y_test,  yproba)
     auc = roc_auc_score(y_test, yproba)
     result_table = result_table.append({'classifiers':key,
                                         "identifier":str(key)[0:-2],
                                         "model_type":str(key)[-2:],
                                         "file_name":key,
                                         'fpr':fpr, 
                                         'tpr':tpr, 
                                         'auc':auc}, ignore_index=True)
   
# result_table.set_index('classifiers', inplace=True)


#%%
count = 1
for identifier in result_table.identifier.unique():
    fig = plt.figure(figsize=(8,8))  
    result_table_segmented = result_table[result_table.identifier.eq(identifier)]
    for i in result_table_segmented.index:
        plt.plot(result_table_segmented.loc[i]['fpr'], 
                 result_table_segmented.loc[i]['tpr'], 
                 label="{}, AUC={:.3f}".format(result_table_segmented.loc[i]["model_type"], result_table_segmented.loc[i]['auc']))
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')
    
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("1-Specificity", fontsize=20)
    
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Sensitivity", fontsize=20)
    
    plt.title(identifier, fontweight='bold', fontsize=30)
    plt.legend(prop={'size':13}, loc='lower right')
    save_name = "test" + str(count)
    fig.savefig(fname=save_name + ".svg",format="svg") # change the figures name here, the 'svg' format is easy to modified by illustrator software.
    plt.show()
    
    count +=1







# In[SHAP feature importance part: suggest use jupyter notebook to finish this part]

# link: https://github.com/slundberg/shap
# conda install -c conda-forge shap
import shap

shap.initjs()

with open(os.path.join(model_path, f"base_model_8.pkl"), 'rb') as f:  # change the model name here
    clf = pickle.load(f)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values, X)


shap.dependence_plot("RM", shap_values, X)

# summarize the effects of all the features
shap.summary_plot(shap_values, X).savefig(fname="shap_importance.svg",format="png")

# In[Explore the core of SHAP]
# Import X, y into jupyter notebook
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

X = pd.read_csv(f"data{suffix}/extracted_features/feature_list.csv", index_col=0)
y = pd.read_csv(f"data{suffix}/extracted_features/label.csv", index_col=0)
total = pd.read_csv(f"data{suffix}/extracted_features/total.csv", index_col=0)
total['label'] =total['label'].astype(int) 

mx = total[total.columns[0:80]] 

my = total['label']

my1 = total["label"].to_numpy()
#model = xgboost.train({"learning_rate": 0.001}, xgboost.DMatrix(X, label=y1), 500)
model = RandomForestClassifier(max_depth=5, random_state=0, n_estimators = 10)

model.fit(mx,my1)
shap_values = shap.TreeExplainer(model).shap_values(mx)
len(shap_values[1][1])

# In[test]
import sklearn
import shap
from sklearn.model_selection import train_test_split

# train a SVM classifier
X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, Y_train)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")


# In[shap with features]
import numpy as np
import pandas as pd
import shap
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
os.chdir("C:\\Users\\hanyu\\Documents\\Duran\\Pre_AI\\20200313_new_anno") # identify the working directionary
suffix = '_Italy_train'
total = pd.read_csv(f"data{suffix}/total_feat.csv", index_col=0)
total = total.drop('sample_id',1)
#clf = RandomForestClassifier(max_features='auto',max_depth=5, n_estimators=100, random_state=0)
#clf.fit(total.drop('label',1),total['label'].values)


model = RandomForestClassifier(n_jobs=-1, max_features='auto',
                                         n_estimators=100, max_depth=5,
                                         random_state=0)
model.fit(total.drop('label',1),total['label'].values)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(total.drop('label',1))
shp = np.mean(np.abs(shap_values[1][:231,:82]),axis=0)  # axis 0 is for features
len(shp)
shplst = shp.tolist()
#total.drop('label',1)
#total.head(10)

'''
shpp = {
    'shapvalue':shplst,
    'feature':lst
}
shpdf = pd.DataFrame(shp,columns=['shapvalue'])
#shpdf.to_csv(shpdf, "shpdf.csv")
'''
shpdf = pd.DataFrame(shp,columns=['shapvalue'])
shpdf.head()
shpdf.to_csv("shpdf1.csv")
