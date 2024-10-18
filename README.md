# Attention-based-isolation-forest

# Changes

 - updated requirements.txt
 - preprocessing for our dataset in atif/datasets/dataset_anomaly.py
 - dataset config in configs/dataset/anomaly.yaml
 - set relative paths

# file paths:

 - configs/logger.yaml
 - configs/dataset/anomaly.yaml
 - configs/optimization.yaml
 - datasets/dataset_anomaly.py

# run:

## in configs/model/attention_based_isolation_forest_sklearn.yaml:

 - setting mode to 0: basic iForest
 - setting mode to 1: attention-based iForest
 - set parameters like n_estimators here 

## in configs/config.yaml:

 - run in optimzation to find ideal parameters
 - run in inference to get output
