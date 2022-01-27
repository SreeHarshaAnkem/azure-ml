from azureml.core import Run, Workspace, Dataset, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import argparse
import pickle

run = Run.get_context()
parser = argparse.ArgumentParser()
parser.add_argument("--train-ds-name", dest="train_ds_name", type=str)
parser.add_argument("--test-ds-name", dest="test_ds_name", type=str)
parser.add_argument("--train-out-folder", dest="train_out_folder", type=str)
parser.add_argument("--test-out-folder", dest="test_out_folder", type=str)

args=parser.parse_args()


FEATURES = ['fixed acidity', 'volatile acidity', 
            'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 
            'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
LABEL = "quality"
WS = run.experiment.workspace

def read_data():
    train_df = Dataset.get_by_name(workspace=WS, name=args.train_ds_name).to_pandas_dataframe()
    test_df = Dataset.get_by_name(workspace=WS, name=args.test_ds_name).to_pandas_dataframe()
    return train_df, test_df

def save_as_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def prepare_data():
    df_train, df_test = read_data()
    scaler = MinMaxScaler()
    df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
    df_test[FEATURES] = scaler.transform(df_test[FEATURES])
    train_save_path = os.path.join(args.train_out_folder, "wine-quality-train-prepped.csv")
    df_train.to_csv(train_save_path)
    test_save_path = os.path.join(args.test_out_folder, "wine-quality-test-prepped.csv")
    df_test.to_csv(test_save_path)

    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "scaler.pkl")
    save_as_pickle(path=model_path, obj=scaler)
    model = Model.register(workspace=WS, model_name="wine-quality-scaler", 
                   model_path=model_path,
                   description="lr model for wine quality",
                   tags = {"dataset": "wine_train"}
                  )

prepare_data()
run.complete()
