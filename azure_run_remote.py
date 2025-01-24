import sys
import argparse
from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Submit an Azure ML job.")
parser.add_argument("--ct", type=str, required=True, help="Compute target (e.g., GPU-A100-Single).")
parser.add_argument("--jobname", type=str, required=True, help="Name of the job script to run.")
parser.add_argument("--experiment_name", type=str, required=True, help="Name of the Azure ML experiment.")
parser.add_argument("--script_args", nargs=argparse.REMAINDER, help="Additional arguments for the script.")
args = parser.parse_args()

# Initialize Azure ML workspace
ws = Workspace.from_config()

# Get environment
env = Environment.get(ws, "corebehrt-phair-sdkv1", version="13")

# Create ScriptRunConfig
src = ScriptRunConfig(
    source_directory=".",
    script=args.jobname,
    compute_target=args.ct,
    environment=env,
    arguments=args.script_args
)

# Submit experiment
experiment = Experiment(ws, name=args.experiment_name)
run = experiment.submit(src)

print(f"Submitted job '{args.jobname}' to compute target '{args.ct}' with experiment name '{args.experiment_name}'.")
# print(f"Experiment URL: {run.get_portal_url()}")