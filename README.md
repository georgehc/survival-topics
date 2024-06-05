# Neural Topic Models with Survival Supervision

A neural network approach that jointly learns a survival model, which predicts time-to-event outcomes, and a topic model, which captures how features relate.

This code accompanies the paper:

> George H. Chen\*, Linhong Li\*, Ren Zuo, Amanda Coston, Jeremy C. Weiss. "Neural Topic Models with Survival Supervision: Jointly Predicting Time-to-Event Outcomes and Learning How Clinical Features Relate".\
> \[[arXiv](https://arxiv.org/abs/2007.07796)\]

\* denotes equal contribution

The code in this repository is primarily by Linhong Lexie Li, George H. Chen, and Ren Zuo.

Importantly, two files in this repository `supervised_topic_models/scholar.py` and `supervised_topic_models/sparse_scholar.py` are primarily *not* by us and are instead modified versions of the [Scholar code by Dallas Card](https://github.com/dallascard/scholar). These two files are under an Apache 2.0 license, following the license of the original code by Dallas Card.

Meanwhile, the three files `baselines/nonparametric_survival_models.py`, `baselines/random_survival_forest_cython.pyx`, and `setup_random_survival_forest_cython.py` are from George H. Chen's earlier repository [npsurvival](https://github.com/georgehc/npsurvival); these files are under an MIT license.

The rest of our code is under an MIT license. For details, see the [LICENSE](LICENSE) file.

## Table of Contents

* [Models](#models)
* [Datasets](#datasets)
* [Topics Learned](#topics-learned-model-outputs)
* [Running Experiments](#running-experiments)
  * [Requirements](#required-packages)
  * [Tutorial](#running-experiments)
    * [Demo](#demo-training-scholar-lda-cox-on-the-support-cancer-dataset)
  * [Visualizing Topics Learned](#generating-topic-heatmaps)
* [References](#references)
  
## Models

We have the following models implemented in our repository. Each model is linked to its implementation script. These models may take in data in different formats, as documented in the **Data Format** column.

| Model  | Description | Source | Data Format&dagger; |
| ------ | ----------- | ---- | ----------- |
| [cox](baselines/cox.py) | Cox regression with elastic-net regularization (wrapper for [`glmnet_python`](https://hastie.su.domains/glmnet_python/)) | \[[Simon et al., 2011](#simon2011regularization) | cox |
| [knnkm](baselines/knnkm.py) | k-nearest-neighbor Kaplan-Meier estimator | \[[Beran, 1981](#beran1981nonparametric)\] | original |
| [rsf](baselines/rsf.py) | random survival forest | \[[Ishwaran et al., 2008](#ishwaran2008random)\] | original |
| [deepsurv](baselines/deepsurv.py) | DeepSurv ([PyCox](https://github.com/havakv/pycox) implementation) | \[[Katzman et al., 2018](#katzman2018deepsurv)\] | original |
| [deephit](baselines/deephit.py) | DeepHit ([PyCox](https://github.com/havakv/pycox) implementation) | \[[Lee et al., 2018](#lee2018deephit)\] | original |
| [naive lda-cox](supervised_topic_models/naive_ldacox.py) | LDA topic model naively combined with Cox regression (topic model learned first in unsupervised fashion, and then the survival model is learned using different data points' topic vectors as input) | N/A | discretized |
| [scholar lda-cox/lda-aft/sage-cox/sage-aft](supervised_topic_models/scholar.py) | [scholar](https://github.com/dallascard/scholar) (LDA or SAGE topic models) combined with either Cox regression or an AFT model (topic and survival models are *jointly* learned) | our paper | discretized |
| [sparse scholar lda-cox/lda-aft/sage-cox/sage-aft](supervised_topic_models/sparse_scholar.py) | *experimental* sparse version of the scholar sage-cox/sage-aft models (note: LDA is not currently supported here; from our preliminary experiments thus far, we have found that encouraging sparsity results in less interpretable models) | our paper | discretized |

&dagger; There are three data formats mentioned. The topic models we use require all features to be discretized (hence why the data format is listed as "discretized" for these models); in particular, continuous variables must be discretized prior to model training. Models that use the "original" features still require that we encode features that are originally discrete using, for instance, one-hot encoding. For the Cox model, each discrete variable is encoded in a manner where we leave out one of the categories (we one-hot encode and then drop the very first one-hot-encoded feature); the category left out is treated as the baseline that the other categories of the variable are measured against (if we did not leave one category out, the unregularized Cox model is known to run into a numerical issue due to collinearity).

Details on the experimental setup are in Section 4 of our paper. Hyperparameter grids are in Appendix B of our paper.

## Datasets

Our experiments used the following datasets:

| Dataset  | Descriptions | # Subjects | # Features | % Censored |
| -------- | ------------ | ---------- | ---------- | ---------- |
| SUPPORT1&Dagger; | acute resp. failure/multiple organ sys. failure | 4194 | 14 | 35.6% |
| SUPPORT2&Dagger; | COPD/congestive heart failure/cirrhosis | 2804 | 14 | 38.8% |
| SUPPORT3&Dagger; | cancer | 1340 | 13 | 11.3% |
| SUPPORT4&Dagger; | coma | 591 | 14 | 18.6% |
| METABRIC \[[Curtis et al., 2012](#curtis2012genomic)\] | breast cancer | 1981 | 24 | 55.2% |
| UNOS&sect; | heart transplantation | 62644 | 49 | 50.2% |
| MIMIC-ICH \[[Johnson et al., 2016a](#johnson2016mimic-a),[b](#johnson2016mimic-b)\] | intracerebral hemorrhage | 961 | 1530 | 23.1% |

&Dagger; SUPPORT1/SUPPORT2/SUPPORT3/SUPPORT4 are disjoint subsets of the same underlying publicly available dataset called SUPPORT \[[Knaus et al., 1995](#knaus1995support)\]; specifically, the different subsets correspond to different disease groups and are chosen to be the same subsets used by [Harrell [2015]](#harrell2015regression)
<br>
&sect; We use the UNOS Standard Transplant and Analysis Research data from the Organ Procurement and Transplantation Network as of September 2019, requested at: [https://www.unos.org/data/](https://www.unos.org/data/)

Note that all datasets are on predicting time until death, except for MIMIC-ICH, for which we predict ICU length of stay.

Dataset and preprocessing details are in Appendix A of our paper.

Note that for the public release of our code, we have only included the SUPPORT1/SUPPORT2/SUPPORT3/SUPPORT4 datasets because these are derived from the publicly available SUPPORT dataset, which can be obtained from [https://hbiostat.org/data/](https://hbiostat.org/data/) courtesy of the Vanderbilt University Department of Biostatistics. The other datasets (METABRIC, UNOS, MIMIC-ICH) require applying for access; while we do not provide these other datasets, we do provide final trained models for them, from which one can obtain topic heatmap visualizations.

## Topics Learned (Model Outputs)

Topics learned by our proposed approach can be visualized as heatmaps. Below is an example for the **SUPPORT-3** (cancer subset of SUPPORT) dataset. To see all heatmap visualizations from our already trained models, see [here](example_plots/). Refer to the paper for how to interpret these heatmaps.

![](example_heatmap.png)

## Running Experiments

We now explain how to train the models from our paper. Note that due to the inherent randomness in model training (which could also depend on software versions and hardware), you might not get exactly the same model parameters as what we got. We have included our pre-trained models in the directory `./example_output/` for reference (these are the same ones used to create visualizations for our paper). Note that since we have trained the models from our paper, we have added new functionality to our code that aims to make reproducibility easier although we currently suspect that there can still be differences in outputs across software versions and across different machines.

### Required Packages

We used Anaconda Python 3. Specific package requirements could be found [here](requirements.txt). You could set up the required environment in the command line:

```
>>> python -m venv env_survivaltopics
>>> source env_survivaltopics/bin/activate
>>> pip install -r requirements.txt
```

After ensuring that all the requirements above installed, if you would like to use the provided random survival forest implementation (which is from George H. Chen's earlier [npsurvival](https://github.com/georgehc/npsurvival), then you will need to compile Cython code:

```
>>> cd baselines
>>> python setup_random_survival_forest_cython.py build_ext --inplace
```

### Running Experiments

To run an experiment:

1. `git clone` this repo to a local directory.
2. Make sure all required packages are installed (see section **Required Packages**).
3. `cd` into the repo directory, replace the `dataset/` folder with one that actually contains the data. Data is omitted in this repo because some of our datasets require applying for access.
4. Make sure hyperparameter search boxes are configured in a `json` file under `configurations/`. You could find plenty of examples [here](configurations/).
5. Modify experiment settings in the bash script `run_experiments.sh`, and type `bash run_experiments.sh` in the command line.
6. This will kick off the experiment. Be sure to name experiments properly using the `experiment_id` option, and note that rerunning using the same `experiment_id` will erase saved outputs from the last experiment with the same `experiment_id`.

#### Demo: Training Scholar LDA-Cox on the SUPPORT\_Cancer Dataset

Follow this demo to see how experiments are configured.

1. For all experiments, hyperparameter search configuration should be specified using a `json` file under `configurations/`. The `json` file's naming convention should follow `model-suffix_identifier.json`. For this demo, we use [`scholar_ldacox-vanilla.json`](configurations/scholar_ldacox-vanilla.json):

```
{"params": {"embedding_dim": [16, 32, 64], "survival_loss_weight": [1.0, 100.0, 10000.0, 1000000.0], "n_topics": [2, 3, 4, 5, 6], "batch_size": [256], "learning_rate": [0.01, 0.001]}}
```

The above configuration specifies the hyperparameter grid. In this case, it corresponds to doing grid search over the hyperparameters `embedding_dim`, `survival_loss_weight`, `n_topics`, `batch_size`, and `learning_rate`. For `embedding_dim`, we search over the values 16, 32, 64, etc.

2. Modify settings in the bash script to specify which dataset and model to use, name the experiment, and specify whether a previously trained model should be loaded etc. Details are documented in the bash script [`run_experiments.sh`](run_experiments.sh). For example, if you only wanted to train the Scholar LDA-Cox model on the SUPPORT\_Cancer dataset, then you can run a bash script that contains the following:

```
dataset=SUPPORT_Cancer
model=scholar_ldacox
n_outer_iter=1
tuning_scheme=grid
tuning_config=vanilla    # this will locate the configuration json file to be ./configurations/scholar_ldacox-vanilla.json
log_dir=output           # directory where experiment outputs are saved
experiment_id=demo_bootstrap_predictions_explain
saved_experiment_id=None
readme_msg=EnterAnyMessageHere
preset_dir=None
manual_train_test=1      # whether to use manually specified train/test splits; this is for reproducing the numerical scores in our paper's tables (1 = yes, 0 = no)

mkdir -p ${log_dir}/${dataset}/${model}/${experiment_id}

python experiments.py ${dataset} ${model} ${n_outer_iter} ${tuning_scheme} ${tuning_config} ${experiment_id} ${saved_experiment_id} ${readme_msg} ${preset_dir} ${manual_train_test} --log_dir ${log_dir}
```

Experiment outputs will be saved to `${log_dir}/${dataset}/${model}/${experiment_id}/`. For this demo, this evaluates to `output/SUPPORT_Cancer/scholar_ldacox/demo_bootstrap_predictions_explain/`.

As mentioned previously, we have included our already trained model output in the directory `./example_output/`. You can see the recorded transcript of the training procedure [here](example_output/SUPPORT_Cancer/scholar_ldacox/demo_bootstrap_predictions_explain/transcript.txt), where the test set time-dependent concordance index is **0.5693679383054495** on the test set (95% confidence interval **(0.53270426, 0.60834197)**).

### Generating Topic Heatmaps

See [this notebook](topic_heatmaps.ipynb) for how to obtain the heatmaps from our paper. Running this notebook requires you to specify a directory that contains the saved model outputs, which is usually `${log_dir}/${dataset}/${model}/${experiment_id}/`. Right now by default the script outputs plots to `./plots/`; we have already run this and stored the example output plots for our saved models to `./example_plots/`.

## References

<a id="beran1981nonparametric"></a>
R. Beran. Nonparametric regression with randomly censored survival data. Technical report, University of California, Berkeley, 1981.

<a id="curtis2012genomic"></a>
C. Curtis,  S. P. Shah,  S.-F. Chin, G. Turashvili, O. M. Rueda, M. J. Dunning, D. Speed, A. G. Lynch, S. Samarajiwa, Y. Yuan, S. Graf, G. Ha, G. Haffari, A. Bashashati, R. Russell, S. McKinney, METABRIC Group, A. Langerod, A. Green, E. Provenzano, G. Wishart, S. Pinder, P. Watson, F. Markowetz, L. Murphy, I. Ellis, A. Purushotham, A.-L. Borresen-Dale, J. D. Brenton, S. Tavare, C. Caldas, and S. Aparicio. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. Nature, 486(7403):346, 2012.

<a id="harrell2015regression"></a> 
F. E. Harrell Jr. Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis. Springer, 2015.

<a id="ishwaran2008random"></a>
Hemant Ishwaran, Udaya B. Kogalur, Eugene H. Blackstone, and Michael S. Lauer. Random survival forests. The Annals of Applied Statistics, 2(3):841–860, 2008.

<a id="johnson2016mimic-a"></a> 
A. E. Johnson, T. J. Pollard, and R. G. Mark. MIMIC-III Clinical Database Demo (version 1.4). PhysioNet, 2016a.

<a id="johnson2016mimic-b"></a> 
A. E. Johnson, T. J. Pollard, L. Shen, L. H. Lehman, M. Feng, M. Ghassemi, B. Moody,  P. Szolovits, L. A. Celi, and R. G. Mark. MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 2016b.

<a id="katzman2018deepsurv"></a>
J. L. Katzman, U. Shaham, A. Cloninger, J. Bates, T. Jiang, and Y. Kluger. DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Medical Research Methodology, 18(1):24, 2018.

<a id="knaus1995support"></a> 
W. A. Knaus, F. E. Harrell, J. Lynn, L. Goldman, R. S. Phillips, A. F. Connors, N. V. Dawson, W. J. Fulkerson, R. M. Califf, N. Desbiens, P. Layde, R. K. Oye, P. E. Bellamy, R. B. Hakim, D. P. Wagner. The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults. Annals of Internal Medicine, 122(3):191–203, 1995.

<a id="lee2018deephit"></a>
C. Lee, W. R. Zame, J. Yoon, and M. van der Schaar. DeepHit: A deep learning approach to survival analysis with competing risks. In AAAI Conference on Artificial Intelligence, 2018.

<a id="simon2011regularization"></a>
N. Simon, J. Friedman, T. Hastie, and R. Tibshirani. Regularization paths for Cox's proportional hazards model via coordinate descent. Journal of Statistical Software, 2011.
