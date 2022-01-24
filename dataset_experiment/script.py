from azureml.core import Run, Dataset
import argparse

run = Run.get_context()
parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, dest="dataset_name")
args = parser.parse_args()
ws = run.experiment.workspace
dataset_name = args.dataset_name
dataset = Dataset.get_by_name(name=dataset_name, workspace=ws)
df = dataset.to_pandas_dataframe()

run.log("observation_count", df.shape[0])
run.complete()
