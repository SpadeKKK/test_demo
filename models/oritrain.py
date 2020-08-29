import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve

def oritrain(X, y, dtname, fc):
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
    best_index = {}
    
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
            y_prob_final[i] = np.ones(len(y))*-1
    
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
    
    print("RESULT AVERAGING ACROSS 10-FOLD:")
    
    test_accuracy_scores = [np.mean(np.array(accuracy_scores[i])[:, 1]) for i in accuracy_scores.keys()]
    best_index_LR = np.argmax(test_accuracy_scores[0:2])+1
    best_index_RF = np.argmax(test_accuracy_scores[3:14])+4
    best_index_GB = np.argmax(test_accuracy_scores[15:38])+16
    print(f"\nThe best machine learning model LR is model {best_index_LR}")
    print(f"\nThe best machine learning model RF is model {best_index_RF}")
    print(f"\nThe best machine learning model GB is model {best_index_GB}")
    
    os.makedirs(f"model/sample{fc}{dtname}", exist_ok=True) 
    
    i = 0
    for (model_name, clf) in estimators:
        i += 1
        clf.fit(X, y) # Rerain the models on the whole dataset
        with open(f"model/sample{fc}{dtname}/base_model_{i}.pkl", "wb") as f: # change number
            pickle.dump(clf, f)
    print("\nBase models saved.")
    return best_index_LR, best_index_RF, best_index_GB
