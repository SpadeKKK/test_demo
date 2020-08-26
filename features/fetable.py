import pandas as pd

def fetable(X, fc):
    newX = []
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
    if fc == "s":
        newX = pd.DataFrame(X, columns = a3+ a42)
        return newX
    if fc == "m":
        newX = pd.DataFrame(X, columns = a1 + a22)
        return newX
    if fc == "sm":
        newX = pd.DataFrame(X, columns = a1 + a22 + a3+ a42)
        return newX

    
    