import pickle
import os
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve, log_loss

def validate(Xval, yval, fc, lr, rf, gb):
    test_accuracy = []
    test_logloss = []
    test_sen = []
    test_auc = []
    #model_path = f"model/sample{fc}{dtname}/" # Remember to choose the correct model path
    model_path = f"modeltest/sample{fc}/" # Remember to choose the correct model path
    model_names = [f"base_model_{lr}",f"base_model_{rf}",f"base_model_{gb}"]
    threshold = 0.5
    for model_name in model_names:
        with open(os.path.join(model_path, f"{model_name}.pkl"), 'rb') as f:
            clf = pickle.load(f)
            y_prob = clf.predict_proba(Xval)[:, 1]
            y_pred = (y_prob > threshold).astype('uint8')
            #print('Accuracy', accuracy_score(yval, y_pred))
            test_accuracy.append(accuracy_score(yval, y_pred))
            test_logloss.append(log_loss(yval, y_pred))
            test_sen.append(recall_score(yval, y_pred, pos_label=1))
            test_auc.append(roc_auc_score(yval, y_prob))
    return test_accuracy, test_logloss, test_sen, test_auc