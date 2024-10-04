# Attention-based-isolation-forest

Initial Commit

# Changes

 - updated requirements.txt
 - preprocessing for our dataset in atif/datasets/dataset_anomaly.py
 - dataset config in configs/dataset/anomaly.yaml
 - set relative paths

# file paths:

 - logger.yaml
 - anomaly.yaml

# run:

## in configs/model/attention_based_isolation_forest_sklearn.yaml:

 - setting mode to 0: basic iForest
 - setting mode to 1: attention-based iForest
 - set parameters like n_estimators here 

## in configs/config.yaml:

 - run in optimzation to find ideal parameters
 - run in inference to get output

## in configs/dataset/anomaly.yaml:

 - set path to dataset here