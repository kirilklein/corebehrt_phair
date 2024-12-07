python -m ehr2vec.main.01_create_data
python -m ehr2vec.main.02_pretrain
python -m ehr2vec.main.04_finetune_cv --config_path example_configs/04_finetune_exposure.yaml
python -m ehr2vec.main.05_simulate_binary_outcome
python -m ehr2vec.main.04_finetune_cv --config_path example_configs/05_02_finetune_simulated.yaml
python -m ehr2vec.main.05_predict_counterfactual --config_path example_configs/05_predict_counterfactual.yaml
python -m ehr2vec.main.06_estimate_causal_effect --config_path example_configs/06_estimate_effect_binary.yaml