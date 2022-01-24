from azureml.core import Run
import pandas as pd

run = Run.get_context()
df = pd.read_csv("winequality-white.csv", delimiter=";")
run.log("observation_count", df.shape[0])
run.complete()
