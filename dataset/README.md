# How this directory works

It is possible to supply your own datasets for use with our experimental setup. Per dataset, what you need is a directory named after the dataset, where within the directory, you should have the following files:

- `X.npy`: Feature matrix, where each row is a patient, and each column is a feature
- `Y.npy`: Labels, with a column containing survival times, and a column containing censoring indicators
- `F.txt`: Feature names

Each one of the three data formats ("original", "discretized", and "cox") will have a set of these. Take `X.npy`, for example: our naming convention expects `X.npy`, `X_discretized.npy`, and `X_cox.npy` to contain feature matrices in the formats of "original", "discretized", and "cox" respectively.

To get a sense of how you can supply your own datasets, we have included the data directories for the SUPPORT1, SUPPORT2, SUPPORT3, and SUPPORT4 datasets mentioned in our paper; note that these are stored respectively in the directories `SUPPORT_ARF_MOSF`, `SUPPORT_COPD_CHF_Cirrhosis`, `SUPPORT_Cancer`, and `SUPPORT_Coma`.

In case you are wondering how we created the directories `SUPPORT_ARF_MOSF`, `SUPPORT_COPD_CHF_Cirrhosis`, `SUPPORT_Cancer`, and `SUPPORT_Coma`, we have provided a bash script to generate them (`make_support_directories.sh`).

# Document frequency files

Lastly, within this directory (and not a specific dataset's directory), there are also `doc_freq_*.txt` files. These are used strictly for topic heatmap visualization purposes and store what are called "document frequencies", namely how many patients have specific vocabulary words. For example, the i-th number in `doc_freq_SUPPORT_Cancer.txt` indicates how many patients in the SUPPORT\_Cancer dataset have the i-th vocabulary word in `SUPPORT_Cancer/F_discretized.txt` (this can be determined directly from the `SUPPORT_Cancer/X_discretized.npy` file).
