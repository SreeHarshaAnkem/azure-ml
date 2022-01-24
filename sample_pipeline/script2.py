
from azureml.core import Run
run = Run.get_context()
run.log("message", "running_script_2")
run.complete()
