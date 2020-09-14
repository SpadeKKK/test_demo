import numpy as np

# feature categorical have three: s = statistical, m = morphological, sm = statistical & morphological
def diffeature(data,fc):
    # Select only features of sample in sample_id_list
    sample_id_list = data['sample_id']
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
      
    ## features
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
    
    if fc == "m":
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
    if fc == "s":
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
    if fc == "sm":
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
    
    return X