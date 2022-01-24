
from azureml.core import Run
from azureml.core.compute import AmlCompute, ComputeTarget
run = Run.get_context()
ws = run.experiment.workspace
compute_name="aml-cluster"
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                       min_nodes=0, max_nodes=2,
                                       vm_priority='lowpriority')
aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)
aml_cluster.wait_for_completion(show_output=True)
