# AutoXGB

XGBoost + Optuna:  no brainer

- auto train xgboost directly from CSV files
- auto tune xgboost using optuna
- auto serve best xgboot model using fastapi


Training a model using AutoXGB is a piece of cake. All you need is some tabular data.

```python
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


More details coming soon

NOTE: Not accepting any PRs currently!
