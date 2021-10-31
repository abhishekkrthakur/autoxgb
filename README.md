# AutoXGB

XGBoost + Optuna:  no brainer

- auto train xgboost directly from CSV files
- auto tune xgboost using optuna
- auto serve best xgboot model using fastapi

NOTE: PRs are currently not accepted. If there are issues/problems, please create an issue.

# Installation

Install using pip

    pip install autoxgb


# Usage
Training a model using AutoXGB is a piece of cake. All you need is some tabular data.

## Parameters

```python

###############################################################################
### required parameters
###############################################################################

# path to training data
train_filename = "data_samples/binary_classification.csv"

# path to output folder to store artifacts
output = "output"

###############################################################################
### optional parameters
###############################################################################

# path to test data. if specified, the model will be evaluated on the test data
# and test_predictions.csv will be saved to the output folder
# if not specified, only OOF predictions will be saved
# test_filename = "test.csv"
test_filename = None

# task: classification or regression
# if not specified, the task will be inferred automatically
# task = "classification"
# task = "regression"
task = None

# an id column
# if not specified, the id column will be generated automatically with the name `id`
# idx = "id"
idx = None

# target columns are list of strings
# if not specified, the target column be assumed to be named `target`
# and the problem will be treated as one of: binary classification, multiclass classification,
# or single column regression
# targets = ["target"]
# targets = ["target1", "target2"]
targets = ["income"]

# features columns are list of strings
# if not specified, all columns except `id`, `targets` & `kfold` columns will be used
# features = ["col1", "col2"]
features = None

# categorical_features are list of strings
# if not specified, categorical columns will be inferred automatically
# categorical_features = ["col1", "col2"]
categorical_features = None

# use_gpu is boolean
# if not specified, GPU is not used
# use_gpu = True
# use_gpu = False
use_gpu = True

# number of folds to use for cross-validation
# default is 5
num_folds = 5

# random seed for reproducibility
# default is 42
seed = 42

# number of optuna trials to run
# default is 1000
# num_trials = 1000
num_trials = 100

# time_limit for optuna trials in seconds
# if not specified, timeout is not set and all trials are run
# time_limit = None
time_limit = 360

# if fast is set to True, the hyperparameter tuning will use only one fold
# however, the model will be trained on all folds in the end
# to generate OOF predictions and test predictions
# default is False
# fast = False
fast = False
```

# Python API

To train a new model, you can run:

```python
from autoxgb import AutoXGB


# required parameters:
train_filename = "data_samples/binary_classification.csv"
output = "output"

# optional parameters
test_filename = None
task = None
idx = None
targets = ["income"]
features = None
categorical_features = None
use_gpu = True
num_folds = 5
seed = 42
num_trials = 100
time_limit = 360
fast = False

# Now its time to train the model!
axgb = AutoXGB(
    train_filename=train_filename,
    output=output,
    test_filename=test_filename,
    task=task,
    idx=idx,
    targets=targets,
    features=features,
    categorical_features=categorical_features,
    use_gpu=use_gpu,
    num_folds=num_folds,
    seed=seed,
    num_trials=num_trials,
    time_limit=time_limit,
    fast=fast,
)
axgb.train()
```

# CLI

Train the model using the `autoxgb train` command. The parameters are same as above.

```
autoxgb train \
 --train_filename datasets/30train.csv \
 --output outputs/30days \
 --test_filename datasets/30test.csv \
 --use_gpu
```

You can also serve the trained model using the `autoxgb serve` command.

```bash
autoxgb serve --model_path outputs/mll --host 0.0.0.0 --debug
```

To know more about a command, run:

    `autoxgb <command> --help` 

```
autoxgb train --help


usage: autoxgb <command> [<args>] train [-h] --train_filename TRAIN_FILENAME [--test_filename TEST_FILENAME] --output
                                        OUTPUT [--task {classification,regression}] [--idx IDX] [--targets TARGETS]
                                        [--num_folds NUM_FOLDS] [--features FEATURES] [--use_gpu] [--fast]
                                        [--seed SEED] [--time_limit TIME_LIMIT]

optional arguments:
  -h, --help            show this help message and exit
  --train_filename TRAIN_FILENAME
                        Path to training file
  --test_filename TEST_FILENAME
                        Path to test file
  --output OUTPUT       Path to output directory
  --task {classification,regression}
                        User defined task type
  --idx IDX             ID column
  --targets TARGETS     Target column(s). If there are multiple targets, separate by ';'
  --num_folds NUM_FOLDS
                        Number of folds to use
  --features FEATURES   Features to use, separated by ';'
  --use_gpu             Whether to use GPU for training
  --fast                Whether to use fast mode for tuning params. Only one fold will be used if fast mode is set
  --seed SEED           Random seed
  --time_limit TIME_LIMIT
                        Time limit for optimization
```
