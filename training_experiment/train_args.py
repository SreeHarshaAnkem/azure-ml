from azureml.core import Run, Workspace, Dataset, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle
import argparse
import os

run = Run.get_context()

FEATURES = ['fixed acidity', 'volatile acidity', 
            'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 
            'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
LABEL = "quality"
WS = run.experiment.workspace

parser = argparse.ArgumentParser()
parser.add_argument("--reg-rate", type=float, dest="reg_rate", default=0.01)
args = parser.parse_args()
reg = args.reg_rate
os.makedirs("outputs", exist_ok=True)
def read_data():
    df = pd.read_csv("winequality-white.csv", delimiter=";")
    df["quality"] = np.where(df["quality"]==6, 1, 0)
    df_train, df_test = train_test_split(df, stratify= df["quality"], random_state=9)
    return df_train, df_test

def save_as_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def train():
    train_df, test_df = read_data()
    scaler = MinMaxScaler()
    train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
    test_df[FEATURES] = scaler.transform(test_df[FEATURES])

    save_as_pickle(path="outputs/scaler.pkl", obj=scaler)
    
    lr = LogisticRegression(C=reg)

    lr.fit(train_df[FEATURES], train_df[LABEL])
    train_pred = lr.predict(train_df[FEATURES])
    train_pred_class = np.where(train_pred>0.5, 1,0)
    accuracy = accuracy_score(train_df[LABEL], train_pred_class)
    recall = recall_score(train_df[LABEL], train_pred_class)
    precision = precision_score(train_df[LABEL], train_pred_class)
    train_metrics = {"C":reg,
                     "accurracy": accuracy,
                     "recall":recall,
                     "precision": precision}


    test_pred = lr.predict(test_df[FEATURES])
    test_pred_class = np.where(test_pred>0.5, 1,0)
    accuracy = accuracy_score(test_df[LABEL], test_pred_class)
    recall = recall_score(test_df[LABEL], test_pred_class)
    precision = precision_score(test_df[LABEL], test_pred_class)
    test_metrics = {"C": reg, 
                    "accurracy": accuracy,
                     "recall":recall,
                     "precision": precision}

    run.log_table("train_metrics", train_metrics)
    run.log_table("test_metrics", test_metrics)
    save_as_pickle(path="outputs/model.pkl", obj=lr)

train()
run.complete()
