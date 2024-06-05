set -e

for model in coxph rsf deepsurv deephit naive_ldacox scholar_ldacox scholar_ldadraft scholar_sagecox scholar_sagedraft
do
for dataset in SUPPORT_Coma SUPPORT_Cancer SUPPORT_COPD_CHF_Cirrhosis SUPPORT_ARF_MOSF
do
    n_outer_iter=1         # Number of random train/test splits, this is not the same as number of folds in cross-validation
    tuning_scheme=grid     # Hyperparameter sweeping scheme (currently only "grid" is supported, corresponding to grid search)
    tuning_config=vanilla  # Suffix identifier for configuration json file, which contains hyperparams' search box
    log_dir=output         # Directory to save experiment outputs to (trained models, best parameters, transcripts etc.) (no need to modify)
    experiment_id=demo_bootstrap_predictions_explain  # Should be named as a unique identifier for an experiment, 
                                                      # "bootstrap_predictions" must be in the name to turn on the bootstrapping option, 
                                                      # "explain" must be in the name to turn on the explainer option
    saved_experiment_id=None       # If you would like to load a trained model from a previous experiment, put the previous experiment's experiment_id here
    readme_msg=None                # An option to add comments to this experiment
    preset_dir=None                # Suffix identifier for configuration json file, which contains a preset set of hyperparameters

    manual_train_test=1    # Whether to use manually specified train/test splits; this is for reproducing the numerical scores in our paper's tables (1 = yes, 0 = no)

    mkdir -p ${log_dir}/${dataset}/${model}/${experiment_id}

    python experiments.py ${dataset} ${model} ${n_outer_iter} ${tuning_scheme} ${tuning_config} ${experiment_id} ${saved_experiment_id} ${readme_msg} ${preset_dir} ${manual_train_test} --log_dir ${log_dir}
done
done
