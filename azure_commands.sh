#!/usr/bin/env bash

# Usage: ./azure_commands.sh <command> <config_name> <ct>
# Example: ./azure_commands.sh simulate_outcome mace GPU-A100-Single
# Example: ./azure_commands.sh finetune pain GPU-A100-Single
# Example: ./azure_commands.sh calibrate simulated_pain CPU-16C

COMMAND=$1
CONFIG_NAME=$2
CT=$3
JOBNAME=$COMMAND  # Automatically set jobname to the command

# Skip validation for config_name and ct if the command is 'help'
if [[ "$COMMAND" != "help" ]]; then
  # Check if config_name and ct are provided for commands that require them
  if [[ -z "$CONFIG_NAME" || -z "$CT" ]]; then
    echo "Error: Config name or compute target not provided for command '$COMMAND'."
    echo "Usage: $0 <command> <config_name> <ct>"
    exit 1
  fi
fi

case "$COMMAND" in
  pretrain)
    echo "Running finetune with config: $CONFIG_NAME, compute target: $CT, and jobname: $JOBNAME..."
    python azure_run_remote.py \
      --ct "$CT" \
      --jobname "ehr2vec/main/02_pretrain.py" \
      --experiment_name "$JOBNAME" \
      --script_args --config_path "configs/azure/pretrain/$CONFIG_NAME"
    ;;

  finetune)
    echo "Running finetune with config: $CONFIG_NAME, compute target: $CT, and jobname: $JOBNAME..."
    python azure_run_remote.py \
      --ct "$CT" \
      --jobname "ehr2vec/main/04_finetune_cv.py" \
      --experiment_name "$JOBNAME" \
      --script_args --config_path "configs/azure/finetune/$CONFIG_NAME"
    ;;

  finetune_tab)
    echo "Running finetune on tabular data with config: $CONFIG_NAME, compute target: $CT, and jobname: $JOBNAME..."
    python azure_run_remote.py \
      --ct "$CT" \
      --jobname "ehr2vec/main/05_train_tab_outcome.py" \
      --experiment_name "$JOBNAME" \
      --script_args --config_path "configs/azure/finetune/$CONFIG_NAME"
    ;;

  simulate_outcome)
    echo "Running outcome simulation with config: $CONFIG_NAME, compute target: $CT, and jobname: $JOBNAME..."
    python azure_run_remote.py \
      --ct "$CT" \
      --jobname "ehr2vec/main/05_simulate_binary_outcome.py" \
      --experiment_name "$JOBNAME" \
      --script_args --config_path "configs/azure/simulate/$CONFIG_NAME"
    ;;

  simulate_from_embeddings)
    echo "Running outcome simulation from embeddings with config: $CONFIG_NAME, compute target: $CT, and jobname: $JOBNAME..."
    python azure_run_remote.py \
      --ct "$CT" \
      --jobname "ehr2vec/main/05_simulate_outcome_from_emb.py" \
      --experiment_name "$JOBNAME" \
      --script_args --config_path "configs/azure/simulate/$CONFIG_NAME"
    ;;

  calibrate)
    echo "Running calibration with config: $CONFIG_NAME, compute target: $CT, and jobname: $JOBNAME..."
    python azure_run_remote.py \
      --ct "$CT" \
      --jobname "ehr2vec/main/04_02_calibrate.py" \
      --experiment_name "$JOBNAME" \
      --script_args --config_path "configs/azure/calibrate/$CONFIG_NAME"
    ;;

  predict_cf)
    echo "Running counterfactual prediction with config: $CONFIG_NAME, compute target: $CT, and jobname: $JOBNAME..."
    python azure_run_remote.py \
      --ct "$CT" \
      --jobname "ehr2vec/main/05_predict_counterfactual.py" \
      --experiment_name "$JOBNAME" \
      --script_args --config_path "configs/azure/counterfactuals/$CONFIG_NAME"
    ;;

  estimate)
    echo "Running causal effect estimation with config: $CONFIG_NAME, compute target: $CT, and jobname: $JOBNAME..."
    python azure_run_remote.py \
      --ct "$CT" \
      --jobname "ehr2vec/main/06_estimate_causal_effect.py" \
      --experiment_name "$JOBNAME" \
      --script_args --config_path "configs/azure/estimate/$CONFIG_NAME"
    ;;

  help|*)
    echo "Usage: $0 <command> <config_name> <ct>"
    echo "Commands:"
    echo "  simulate_outcome         - Run outcome simulation"
    echo "  simulate_from_embeddings - Run outcome simulation from embeddings"
    echo "  finetune                 - Run finetune"
    echo "  finetune_tab             - Run finetune on tabular data"
    echo "  calibrate                - Run calibration"
    echo "  predict_cf               - Run counterfactual prediction"
    echo "  estimate                 - Run causal effect estimation"
    echo "Example: $0 finetune mace GPU-A100-Single"
    echo "Example: $0 simulate_outcome pain CPU-16C"
    exit 1
    ;;
esac