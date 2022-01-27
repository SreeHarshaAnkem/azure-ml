from azureml.core import Run, Workspace, Dataset, Model
from azureml.data import OutputFileDatasetConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score
import os
import argparse
import numpy as np
import pandas as pd
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str, dest="train_data")
parser.add_argument("--test-data", type=str, dest="test_data")
args = parser.parse_args()

run = Run.get_context()
WS = run.experiment.workspace

FEATURES = ['fixed acidity', 'volatile acidity', 
            'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 
            'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
LABEL = "quality"
def save_as_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def create_dataframe(path):
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    files = [os.path.join(path, f) for f in files]
    print("files found : {files}")
    df = pd.DataFrame()
    for f in files:
        df = pd.concat([df, pd.read_csv(f)])
    return df
    
def train():
    df_train = create_dataframe(args.train_data)
    df_test = create_dataframe(args.test_data)

    lr = LogisticRegression()
    
    lr.fit(df_train[FEATURES], df_train[LABEL])
    train_pred = lr.predict(df_train[FEATURES])
    train_pred_class = np.where(train_pred>0.5, 1,0)
    accuracy = accuracy_score(df_train[LABEL], train_pred_class)
    recall = recall_score(df_train[LABEL], train_pred_class)
    precision = precision_score(df_train[LABEL], train_pred_class)
    train_metrics = {"accurracy": accuracy,
                     "recall":recall,
                     "precision": precision}
    
    
    test_pred = lr.predict(df_test[FEATURES])
    test_pred_class = np.where(test_pred>0.5, 1,0)
    accuracy = accuracy_score(df_test[LABEL], test_pred_class)
    recall = recall_score(df_test[LABEL], test_pred_class)
    precision = precision_score(df_test[LABEL], test_pred_class)
    test_metrics = {"accurracy": accuracy,
                     "recall":recall,
                     "precision": precision}
    
    run.log_table("train_metrics", train_metrics)
    run.log_table("test_metrics", test_metrics)
    
    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "model.pkl")
    save_as_pickle(path=model_path, obj=lr)
    model = Model.register(workspace=WS, model_name="wine-quality-lr", 
                   model_path=model_path,
                   description="lr model for wine quality",
                   tags = {"dataset": "wine_train"}
                  )
train()
run.complete()
