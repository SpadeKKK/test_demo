import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve, log_loss
import matplotlib.pyplot as plt
def roc(X_train, y_train, X_test, y_test):
    model_path = f"model/sample/" # Remember to choose the correct model path
    model_names = [f"base_model_2",f"base_model_8",f"base_model_16"]
    threshold = 0.5
    classifiers = []
    for model_name in model_names:
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
    #    plt.title(f'{feature}_features in \n {dataset} dataset', fontweight='bold', fontsize=30)
    plt.legend(prop={'size':13}, loc='lower right')
    return fig
