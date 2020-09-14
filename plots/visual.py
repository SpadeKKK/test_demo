import os
import numpy as np
import pandas as pd
import pymzml
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math

def visual(filename, samplename):
    suffix = f'{filename}' # dataset name
    mzml_path = f"data_{suffix}/mzml_data_conversion_SRMspectra/" # folder path to mzml mass spec files
 
    SRM_list_i = ['Transition1(632.29m/z)', 'Transition2(703.33m/z)', 'Transition3(832.37m/z)', 'Transition4(960.43m/z)',
                  'Transition5(1031.47m/z)', 'Transition6(1144.55m/z)', 'Transition7(1245.60m/z)']
    
    SRM_list_t = ['Transition1(622.29m/z)', 'Transition2(693.33m/z)', 'Transition3(822.37m/z)', 'Transition4(950.43m/z)',
                  'Transition5(1021.47m/z)', 'Transition6(1134.55m/z)', 'Transition7(1235.60m/z)']
    def smooth(x):
        return savgol_filter(x, 9, 3)

    sample_id = samplename
    mzml_file = os.path.join(mzml_path, sample_id + '.mzML')
    if os.path.exists(mzml_file):
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
                if np.round(selected_precursor, 2) == 802.37: # Standard
                    standard_scanid.append(spectrum.ID)
                    standard_scantime.append(float(spectrum.get_element_by_path(['scanList', 'scan', 'cvParam'])[0].get('value')))
                    standard_intensity.append(spectrum.i)
                    standard_tic.append(spectrum.TIC)
                if np.round(selected_precursor, 2) == 797.38: # Starget
                    target_scanid.append(spectrum.ID)
                    target_intensity.append(spectrum.i)
                    target_tic.append(spectrum.TIC)
                    
        standard_tic = np.array(standard_tic)
        standard_scantime = np.array(standard_scantime)
        standard_max_index = np.argmax(standard_tic[standard_scantime < 20], axis=0)
        standard_intensity = np.array(standard_intensity)
        target_intensity = np.array(target_intensity)

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
        thr = math.floor(np.std(new)*1.96) # if mean scan is not 0, the boundaries are (-thr, thr)
         
        # Hardcode +/-scan_range around the highest peak of internal standard for now
        scan_range = 20
        lower_bound = standard_max_index-scan_range
        upper_bound = standard_max_index+scan_range
    
        fig_i = plt.figure(figsize=(6,2))
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 standard_intensity[lower_bound: upper_bound, 0], color='k', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 standard_intensity[lower_bound: upper_bound, 1], color='b', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 standard_intensity[lower_bound: upper_bound, 2], color='g', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 standard_intensity[lower_bound: upper_bound, 3], color='r', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 standard_intensity[lower_bound: upper_bound, 4], color='c', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 standard_intensity[lower_bound: upper_bound, 5], color='m', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 standard_intensity[lower_bound: upper_bound, 6], color='y', linewidth=1)
        plt.ylabel('Intensity')
        plt.title('Internal Standard')
        plt.axvspan(standard_scantime[lower_bound+(20-thr)], standard_scantime[upper_bound-(20-thr)], color = 'b', alpha = 0.1, lw = 0)
        lgd = plt.legend(SRM_list_i, loc='center left', bbox_to_anchor=(1, 0.5))
        fig_i.savefig(f"data_{suffix}/" + sample_id + '_standard.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig_i)
        
        fig_t = plt.figure(figsize=(6,2))
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 smooth(target_intensity[:, 0])[lower_bound: upper_bound], color='k', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 smooth(target_intensity[:, 1])[lower_bound: upper_bound], color='b', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 smooth(target_intensity[:, 2])[lower_bound: upper_bound], color='g', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 smooth(target_intensity[:, 3])[lower_bound: upper_bound], color='r', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 smooth(target_intensity[:, 4])[lower_bound: upper_bound], color='c', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 smooth(target_intensity[:, 5])[lower_bound: upper_bound], color='m', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 smooth(target_intensity[:, 6])[lower_bound: upper_bound], color='y', linewidth=1)
        plt.ylabel('Intensity')
        plt.title('Target')
        plt.axvspan(standard_scantime[lower_bound+(20-thr)], standard_scantime[upper_bound-(20-thr)], color = 'b', alpha = 0.1, lw = 0)
        lgd = plt.legend(SRM_list_t, loc='center left', bbox_to_anchor=(1, 0.5))
        fig_t.savefig(f"data_{suffix}/" + sample_id + '_target_smooth.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig_t)
    else: 
        print("file not existed.")
    return fig_i, fig_t


def visualnor(filename, samplename):
    suffix = f'{filename}' # dataset name
    mzml_path = f"data_{suffix}/mzml_data_conversion_SRMspectra/" # folder path to mzml mass spec files
 
    SRM_list_i = ['Transition1(632.29m/z)', 'Transition2(703.33m/z)', 'Transition3(832.37m/z)', 'Transition4(960.43m/z)',
                  'Transition5(1031.47m/z)', 'Transition6(1144.55m/z)', 'Transition7(1245.60m/z)']
    
    SRM_list_t = ['Transition1(622.29m/z)', 'Transition2(693.33m/z)', 'Transition3(822.37m/z)', 'Transition4(950.43m/z)',
                  'Transition5(1021.47m/z)', 'Transition6(1134.55m/z)', 'Transition7(1235.60m/z)']
    def smooth(x):
        return savgol_filter(x, 9, 3)

    sample_id = samplename
    mzml_file = os.path.join(mzml_path, sample_id + '.mzML')
    if os.path.exists(mzml_file):
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
                if np.round(selected_precursor, 2) == 802.37: # Standard
                    standard_scanid.append(spectrum.ID)
                    standard_scantime.append(float(spectrum.get_element_by_path(['scanList', 'scan', 'cvParam'])[0].get('value')))
                    standard_intensity.append(spectrum.i)
                    standard_tic.append(spectrum.TIC)
                if np.round(selected_precursor, 2) == 797.38: # Starget
                    target_scanid.append(spectrum.ID)
                    target_intensity.append(spectrum.i)
                    target_tic.append(spectrum.TIC)
                    
        standard_tic = np.array(standard_tic)
        standard_scantime = np.array(standard_scantime)
        standard_max_index = np.argmax(standard_tic[standard_scantime < 20], axis=0)
        standard_intensity = np.array(standard_intensity)
        target_intensity = np.array(target_intensity)

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
        thr = math.floor(np.std(new)*1.96) # if mean scan is not 0, the boundaries are (-thr, thr)

        def normalize(lst):
            if max(lst) != 0:
                return [(float(i)-min(lst))/(max(lst)-min(lst)) for i in lst]
            else:
                return lst
            
        # Hardcode +/-scan_range around the highest peak of internal standard for now
        scan_range = 20
        lower_bound = standard_max_index-scan_range
        upper_bound = standard_max_index+scan_range
    
        fig_ni = plt.figure(figsize=(6,2))
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(standard_intensity[lower_bound: upper_bound, 0]), color='k', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(standard_intensity[lower_bound: upper_bound, 1]), color='b', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(standard_intensity[lower_bound: upper_bound, 2]), color='g', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(standard_intensity[lower_bound: upper_bound, 3]), color='r', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(standard_intensity[lower_bound: upper_bound, 4]), color='c', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(standard_intensity[lower_bound: upper_bound, 5]), color='m', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(standard_intensity[lower_bound: upper_bound, 6]), color='y', linewidth=1)
        plt.ylabel('Intensity_Normalized')
        plt.title('Internal Standard_Normalized')
        plt.axvspan(standard_scantime[lower_bound+(20-thr)], standard_scantime[upper_bound-(20-thr)], color = 'b', alpha = 0.1, lw = 0)
        lgd = plt.legend(SRM_list_i, loc='center left', bbox_to_anchor=(1, 0.5))
        fig_ni.savefig(f"data_{suffix}/" + sample_id + '_n_standard.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig_ni)
        
        fig_nt = plt.figure(figsize=(6,2))
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(smooth(target_intensity[:, 0])[lower_bound: upper_bound]), color='k', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(smooth(target_intensity[:, 1])[lower_bound: upper_bound]), color='b', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(smooth(target_intensity[:, 2])[lower_bound: upper_bound]), color='g', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(smooth(target_intensity[:, 3])[lower_bound: upper_bound]), color='r', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(smooth(target_intensity[:, 4])[lower_bound: upper_bound]), color='c', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(smooth(target_intensity[:, 5])[lower_bound: upper_bound]), color='m', linewidth=1)
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(smooth(target_intensity[:, 6])[lower_bound: upper_bound]), color='y', linewidth=1)
        plt.ylabel('Intensity_Normalized')
        plt.title('Target_Normalized')
        plt.axvspan(standard_scantime[lower_bound+(20-thr)], standard_scantime[upper_bound-(20-thr)], color = 'b', alpha = 0.1, lw = 0)
        lgd = plt.legend(SRM_list_t, loc='center left', bbox_to_anchor=(1, 0.5))
        fig_nt.savefig(f"data_{suffix}/" + sample_id + '_n_target_smooth.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig_nt)
    else:
        print("file not existed.")
    return fig_ni, fig_nt

def visualsum(filename, samplename):
    suffix = f'{filename}' # dataset name
    mzml_path = f"data_{suffix}/mzml_data_conversion_SRMspectra/" # folder path to mzml mass spec files

    def smooth(x):
        return savgol_filter(x, 9, 3)

    sample_id = samplename
    mzml_file = os.path.join(mzml_path, sample_id + '.mzML')
    if os.path.exists(mzml_file):
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
                if np.round(selected_precursor, 2) == 802.37: # Standard
                    standard_scanid.append(spectrum.ID)
                    standard_scantime.append(float(spectrum.get_element_by_path(['scanList', 'scan', 'cvParam'])[0].get('value')))
                    standard_intensity.append(spectrum.i)
                    standard_tic.append(spectrum.TIC)
                if np.round(selected_precursor, 2) == 797.38: # Starget
                    target_scanid.append(spectrum.ID)
                    target_intensity.append(spectrum.i)
                    target_tic.append(spectrum.TIC)
                    
        standard_tic = np.array(standard_tic)
        standard_scantime = np.array(standard_scantime)
        standard_max_index = np.argmax(standard_tic[standard_scantime < 20], axis=0)
        standard_intensity = np.array(standard_intensity)
        target_intensity = np.array(target_intensity)

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
        thr = math.floor(np.std(new)*1.96) # if mean scan is not 0, the boundaries are (-thr, thr)
         
        # Hardcode +/-scan_range around the highest peak of internal standard for now
        scan_range = 20
        lower_bound = standard_max_index-scan_range
        upper_bound = standard_max_index+scan_range
    
        fig_ti = plt.figure(figsize=(6,2))
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 np.sum(standard_intensity[lower_bound: upper_bound, 0:6],1), color='k', linewidth=1)

        plt.ylabel('Intensity')
        plt.title('Internal Standard_Integrated')
        plt.axvspan(standard_scantime[lower_bound+(20-thr)], standard_scantime[upper_bound-(20-thr)], color = 'b', alpha = 0.1, lw = 0)
        fig_ti.savefig(f"data_{suffix}/" + sample_id + '_total_standard.png')
        plt.close(fig_ti)
        
        fig_tt = plt.figure(figsize=(6,2))
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 smooth(np.sum(target_intensity[lower_bound: upper_bound, 0:6],1)), color='k', linewidth=1)
        plt.ylabel('Intensity')
        plt.title('Target_Integrated')
        plt.axvspan(standard_scantime[lower_bound+(20-thr)], standard_scantime[upper_bound-(20-thr)], color = 'b', alpha = 0.1, lw = 0)
        fig_tt.savefig(f"data_{suffix}/" + sample_id + '_total_target_smooth.png')
        plt.close(fig_tt)
    else: 
        print("file not existed.")
    return fig_ti, fig_tt

def visualsumnor(filename, samplename):
    suffix = f'{filename}' # dataset name
    mzml_path = f"data_{suffix}/mzml_data_conversion_SRMspectra/" # folder path to mzml mass spec files

    def smooth(x):
        return savgol_filter(x, 9, 3)
    def normalize(lst):
        if max(lst) != 0:
            return [(float(i)-min(lst))/(max(lst)-min(lst)) for i in lst]
        else:
            return lst
    sample_id = samplename
    mzml_file = os.path.join(mzml_path, sample_id + '.mzML')
    if os.path.exists(mzml_file):
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
                if np.round(selected_precursor, 2) == 802.37: # Standard
                    standard_scanid.append(spectrum.ID)
                    standard_scantime.append(float(spectrum.get_element_by_path(['scanList', 'scan', 'cvParam'])[0].get('value')))
                    standard_intensity.append(spectrum.i)
                    standard_tic.append(spectrum.TIC)
                if np.round(selected_precursor, 2) == 797.38: # Starget
                    target_scanid.append(spectrum.ID)
                    target_intensity.append(spectrum.i)
                    target_tic.append(spectrum.TIC)
                    
        standard_tic = np.array(standard_tic)
        standard_scantime = np.array(standard_scantime)
        standard_max_index = np.argmax(standard_tic[standard_scantime < 20], axis=0)
        standard_intensity = np.array(standard_intensity)
        target_intensity = np.array(target_intensity)

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
        thr = math.floor(np.std(new)*1.96) # if mean scan is not 0, the boundaries are (-thr, thr)
         
        # Hardcode +/-scan_range around the highest peak of internal standard for now
        scan_range = 20
        lower_bound = standard_max_index-scan_range
        upper_bound = standard_max_index+scan_range
    
        fig_tni = plt.figure(figsize=(6,2))
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(np.sum(standard_intensity[lower_bound: upper_bound, 0:6],1)), color='k', linewidth=1)

        plt.ylabel('Intensity')
        plt.title('Internal Standard_Integrated')
        plt.axvspan(standard_scantime[lower_bound+(20-thr)], standard_scantime[upper_bound-(20-thr)], color = 'b', alpha = 0.1, lw = 0)
        fig_tni.savefig(f"data_{suffix}/" + sample_id + '_n_total_standard.png')
        plt.close(fig_tni)
        
        fig_tnt = plt.figure(figsize=(6,2))
        plt.plot(standard_scantime[lower_bound: upper_bound],
                 normalize(smooth(np.sum(target_intensity[lower_bound: upper_bound, 0:6],1))), color='k', linewidth=1)
        plt.ylabel('Intensity')
        plt.title('Target_Integrated')
        plt.axvspan(standard_scantime[lower_bound+(20-thr)], standard_scantime[upper_bound-(20-thr)], color = 'b', alpha = 0.1, lw = 0)
        fig_tnt.savefig(f"data_{suffix}/" + sample_id + '_n_total_target_smooth.png')
        plt.close(fig_tnt)
    else:
        print("file not existed.")
    return fig_tni, fig_tnt

def visual_all(filename):

    suffix = f'{filename}' # dataset name
    clincal_info_file = f"data_{suffix}/data_{suffix}.csv" # file path to clinical information csv file. None if not available
    mzml_path = f"data_{suffix}/mzml_data_conversion_SRMspectra/" # folder path to mzml mass spec files
    
    
    def smooth(x):
        return savgol_filter(x, 9, 3)
    
    
    clinical_info = pd.read_csv(clincal_info_file)
    clinical_info_selected = clinical_info[(clinical_info['Current criteria']=='TB') | (clinical_info['Current criteria']=='Not TB')] 
    
    label_list = clinical_info_selected['Current criteria'].values
    print(f"Propotion of positive samples: {np.mean(label_list=='TB')}")
    
    # Format sample id, e.g. 31224(1) will become 31224_1
    def format_sample_id(x):
        return str(x).replace(' ', '').replace(')', '').replace('(', '_')
    clinical_info_selected_sample_id = clinical_info_selected['MaskedID'].values
    sample_id_list = [format_sample_id(x) for x in clinical_info_selected_sample_id]
    
    
    # Create directory to store images
    os.makedirs(f"visualization{suffix}/plot", exist_ok=True)
    os.makedirs(f"visualization{suffix}/plot_peak", exist_ok=True)
    
    # Visualize mass spec data. List of SRM M/z: '622', '693', '822', '950', '1021', '1134', '1235'.
    SRM_list = ['622', '693', '822', '950', '1021', '1134', '1235']
    
    # We look for scantime corresponding to the maximum total ion concentration (TIC) of internal standard.
    # HTML file to visualize the total ion concentration over the whole scan period and relative abundance at this scantime
    html = ""
    html += """<table border="1">
      <tr>
        <th>Sample</th>
        <th>Label</th>
        <th>Total area</th>
        <th>Highest peak</th>
        <th>Scan Time at peak</th>
        <th>Std TIC</th>
        <th>Std SRM</th>
        <th>Std SRM Range</th>
        <th>Tar TIC</th>
        <th>Tar SRM</th>
        <th>Tar SRM Range</th>
      </tr>"""
    
    # HTML file to visualize the each ion intensity around this scantime
    html_peak = ""
    html_peak += """<table border="1">
      <tr>
        <th>Sample</th>
        <th>Label</th>
        <th>Total area</th>
        <th>Highest peak</th>
        <th>Scan Time at peak</th>
        <th>Standard</th>
        <th>Target</th>
        <th>Target Smoothing</th>
      </tr>"""
    
    for i in range(len(sample_id_list)):
        sample_id = sample_id_list[i]
        mzml_file = os.path.join(mzml_path, sample_id + '.mzML')
        if os.path.exists(mzml_file):
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
                    if np.round(selected_precursor, 2) == 802.37: # Standard
                        standard_scanid.append(spectrum.ID)
                        standard_scantime.append(float(spectrum.get_element_by_path(['scanList', 'scan', 'cvParam'])[0].get('value')))
                        standard_intensity.append(spectrum.i)
                        standard_tic.append(spectrum.TIC)
                    if np.round(selected_precursor, 2) == 797.38: # Starget
                        target_scanid.append(spectrum.ID)
                        target_intensity.append(spectrum.i)
                        target_tic.append(spectrum.TIC)
                        
            standard_tic = np.array(standard_tic)
            standard_scantime = np.array(standard_scantime)
            standard_max_index = np.argmax(standard_tic[standard_scantime < 20], axis=0)
            standard_intensity = np.array(standard_intensity)
            target_intensity = np.array(target_intensity)
            
            # Hardcode +/-scan_range around the highest peak of internal standard for now
            scan_range = 20
            lower_bound = standard_max_index-scan_range
            upper_bound = standard_max_index+scan_range
            
            # Save images for Internal Standard
            fig = plt.figure(figsize=(6,2))
            plt.plot(standard_scantime, standard_tic, color='k', linewidth=1)
            plt.ylabel('Intensity')
            plt.title('Internal Standard TIC')
            fig.savefig(f"visualization{suffix}/plot/" + sample_id + '_standard_tic.png')
            plt.close(fig)
            
            fig = plt.figure(figsize=(6,2))
            plt.bar(np.arange(len(SRM_list)), standard_intensity[standard_max_index],
                    align='center', alpha=0.5)
            plt.xticks(np.arange(len(SRM_list)), SRM_list)
            plt.ylabel('Intensity')
            plt.title('Internal Standard SRM')
            fig.savefig(f"visualization{suffix}/plot/" + sample_id + '_standard_srm.png')
            plt.close(fig)
            
            fig = plt.figure(figsize=(6,2))
            plt.bar(np.arange(len(SRM_list)), np.sum(standard_intensity[standard_max_index-scan_range:
                                                                       standard_max_index+scan_range], axis=0),
                    align='center', alpha=0.5)
            plt.xticks(np.arange(len(SRM_list)), SRM_list)
            plt.ylabel('Intensity')
            plt.title('Internal Standard SRM')
            fig.savefig(f"visualization{suffix}/plot/" + sample_id + '_standard_srm_range.png')
            plt.close(fig)
            
            fig = plt.figure(figsize=(6,2))
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     standard_intensity[lower_bound: upper_bound, 0], color='k', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     standard_intensity[lower_bound: upper_bound, 1], color='b', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     standard_intensity[lower_bound: upper_bound, 2], color='g', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     standard_intensity[lower_bound: upper_bound, 3], color='r', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     standard_intensity[lower_bound: upper_bound, 4], color='c', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     standard_intensity[lower_bound: upper_bound, 5], color='m', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     standard_intensity[lower_bound: upper_bound, 6], color='y', linewidth=1)
            plt.ylabel('Intensity')
            plt.title('Internal Standard')
            lgd = plt.legend(SRM_list, loc='center left', bbox_to_anchor=(1, 0.5))
            fig.savefig(f"visualization{suffix}/plot_peak/" + sample_id + '_standard.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close(fig)
            
            # Save images for Target
            fig = plt.figure(figsize=(6,2))
            plt.plot(standard_scantime, target_tic, color='k', linewidth=1)
            plt.ylabel('Intensity')
            plt.title('Target TIC')
            fig.savefig(f"visualization{suffix}/plot/" + sample_id + '_target_tic.png')
            plt.close(fig)
            
            fig = plt.figure(figsize=(6,2))
            plt.bar(np.arange(len(SRM_list)), target_intensity[standard_max_index],
                    align='center', alpha=0.5)
            plt.xticks(np.arange(len(SRM_list)), SRM_list)
            plt.ylabel('Intensity')
            plt.title('Target SRM')
            fig.savefig(f"visualization{suffix}/plot/" + sample_id + '_target_srm.png')
            plt.close(fig)
            
            fig = plt.figure(figsize=(6,2))
            plt.bar(np.arange(len(SRM_list)), np.sum(target_intensity[standard_max_index-scan_range:
                                                                       standard_max_index+scan_range], axis=0),
                    align='center', alpha=0.5)
            plt.xticks(np.arange(len(SRM_list)), SRM_list)
            plt.ylabel('Intensity')
            plt.title('Internal Standard SRM')
            fig.savefig(f"visualization{suffix}/plot/" + sample_id + '_target_srm_range.png')
            plt.close(fig)
            
            fig = plt.figure(figsize=(6,2))
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     target_intensity[lower_bound: upper_bound, 0], color='k', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     target_intensity[lower_bound: upper_bound, 1], color='b', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     target_intensity[lower_bound: upper_bound, 2], color='g', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     target_intensity[lower_bound: upper_bound, 3], color='r', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     target_intensity[lower_bound: upper_bound, 4], color='c', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     target_intensity[lower_bound: upper_bound, 5], color='m', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     target_intensity[lower_bound: upper_bound, 6], color='y', linewidth=1)
            plt.ylabel('Intensity')
            plt.title('Target')
            lgd = plt.legend(SRM_list, loc='center left', bbox_to_anchor=(1, 0.5))
            fig.savefig(f"visualization{suffix}/plot_peak/" + sample_id + '_target.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close(fig)
            
            fig = plt.figure(figsize=(6,2))
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     smooth(target_intensity[:, 0])[lower_bound: upper_bound], color='k', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     smooth(target_intensity[:, 1])[lower_bound: upper_bound], color='b', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     smooth(target_intensity[:, 2])[lower_bound: upper_bound], color='g', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     smooth(target_intensity[:, 3])[lower_bound: upper_bound], color='r', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     smooth(target_intensity[:, 4])[lower_bound: upper_bound], color='c', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     smooth(target_intensity[:, 5])[lower_bound: upper_bound], color='m', linewidth=1)
            plt.plot(standard_scantime[lower_bound: upper_bound],
                     smooth(target_intensity[:, 6])[lower_bound: upper_bound], color='y', linewidth=1)
            plt.ylabel('Intensity')
            plt.title('Target')
            lgd = plt.legend(SRM_list, loc='center left', bbox_to_anchor=(1, 0.5))
            fig.savefig(f"visualization{suffix}/plot_peak/" + sample_id + '_target_smooth.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close(fig)
            
            # Add visualization of TIC and relative abundance to HTML
            if (label_list is not None) and (label_list[i] == 'Confirmed TB'):
                tr_str = "<tr bgcolor='#FF0000'>"
            else:
                tr_str = "<tr>"
            if label_list is not None:
                label = label_list[i]
            else:
                label = ""
            html += tr_str
            html += "<td>" + sample_id + "</td>"
            html += "<td>" + label + "</td>"
            html += "<td>" + str(round(np.sum(standard_tic), 4)) + "</td>"
            html += "<td>" + str(round(standard_tic[standard_max_index])) + "</td>"
            html += "<td>" + str(round(standard_scantime[standard_max_index], 4)) + "</td>"
            html += f"<td><img src='plot/" + sample_id + "_standard_tic.png'/></td>"
            html += f"<td><img src='plot/" + sample_id + "_standard_srm.png'/></td>"
            html += f"<td><img src='plot/" + sample_id + "_standard_srm_range.png'/></td>"
            html += f"<td><img src='plot/" + sample_id + "_target_tic.png'/></td>"
            html += f"<td><img src='plot/" + sample_id + "_target_srm.png'/></td>"
            html += f"<td><img src='plot/" + sample_id + "_target_srm_range.png'/></td>"
            html += "</tr>"
            
            # Add visualization of peak to HTML
            if (label_list is not None) and (label_list[i] == 'Confirmed TB'):
                tr_str = "<tr bgcolor='#FF0000'>"
            else:
                tr_str = "<tr>"
            if label_list is not None:
                label = label_list[i]
            else:
                label = ""
            html_peak += tr_str
            html_peak += "<td>" + sample_id + "</td>"
            html_peak += "<td>" + label + "</td>"
            html_peak += "<td>" + str(round(np.sum(standard_tic), 4)) + "</td>"
            html_peak += "<td>" + str(round(standard_tic[standard_max_index])) + "</td>"
            html_peak += "<td>" + str(round(standard_scantime[standard_max_index], 4)) + "</td>"
            html_peak += f"<td><img src='plot_peak/" + sample_id + "_standard.png'/></td>"
            html_peak += f"<td><img src='plot_peak/" + sample_id + "_target.png'/></td>"
            html_peak += f"<td><img src='plot_peak/" + sample_id + "_target_smooth.png'/></td>"
            html_peak += "</tr>"
        else:
            print(f"{sample_id} does not exist.")
    
    html += "</table>"
    
    with open(f"visualization{suffix}/plot.html", "w") as file:
        file.write(html)
    with open(f"visualization{suffix}/plot_peak.html", "w") as file:
        file.write(html_peak)
    print("All plots finished.")
