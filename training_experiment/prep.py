from azureml.core import Run, Workspace, Dataset, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
run = Run.get_context()

FEATURES = ['fixed acidity', 'volatile acidity', 
            'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 
            'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
LABEL = "quality"
WS = run.experiment.workspace

def read_data():
    df = pd.read_csv("winequality-white.csv", delimiter=";")
    df["quality"] = np.where(df["quality"]==6, 1, 0)
    df_train, df_test = train_test_split(df, stratify= df["quality"], random_state=9)
    return df_train, df_test

def save_as_pickle(path, obj):
    with open(path, "wb") as pkl:
        pickle.dump(obj)

def prepare_data():
    df_train, df_test = read_data()
    scaler = MinMaxScaler()
    df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
    df_test[FEATURES] = scaler.transform(df_test[FEATURES])
    train_ds = Dataset.from_pandas_dataframe(df_train)
    train_ds = Dataset.register(train_ds, workspace=WS, name="wine_train", 
                    description="training_data_wine", 
                    tags={"version":"1", "algo":"logistic"})
    test_ds = Dataset.from_pandas_dataframe(df_test)
    test_ds = Dataset.register(test_ds, workspace=WS, name="wine_test", 
                    description="test_data_wine", 
                    tags={"version":"1", "algo":"logistic"})
    save_as_pickle(path="scaler.pkl", obj=scaler)
    

def train():
    train_ds = Dataset.get_by_name(workspace=WS, name="wine_train")
    train_df = train_ds.to_pandas_dataframe()
    
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
    save_as_pickle(path="model.pkl", obj=lr)
    model = Model.register(workspace=WS, model_name="wine-quality-lr", 
                   model_path="wine-quality-lr.pkl",
                   description="lr model for wine quality",
                   tags = {"dataset": "wine_train"}
                  )
prepare_data()
train()
run.wait_for_completion()
