import numpy as np
import pandas as pd
def dataset(info_file):
    if info_file is not None:
        # 1. Read in patient and sample id from info file.
        info = pd.read_csv(info_file)
        info_selected = info[(info['Current criteria']=='TB') | (info['Current criteria']=='Not TB')] 
    
        label_list = info_selected['Current criteria'].values
        print(f"Propotion of positive samples: {np.mean(label_list=='TB')}")
    
        label_list = (label_list == 'TB').astype('uint8') # Convert label 'Confirmed TB' and 'Unlikely TB' to 1 and 0
    
        def format_sample_id(x):
            return str(x).replace(' ', '').replace(')', '').replace('(', '_')
        info_selected_sample_id = info_selected['Sample ID'].values
        sample_id_list = [x for x in info_selected_sample_id]
    else:
        label_list = None
    return label_list
