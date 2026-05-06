import os 
import dill
import sys 

import numpy as np
import pandas as pd 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score ,precision_score, recall_score

from src.exception import CustomException
from src.logger import logging

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        model = models
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        report['train_accuracy'] = train_acc
        report['test_accuracy'] = test_acc

        return report

    except Exception as e:
        raise CustomException(e, sys)