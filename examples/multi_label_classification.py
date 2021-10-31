from autoxgb import AutoXGB


# required parameters:
train_filename = "data_samples/multi_label_classification.csv"
output = "output"

# optional parameters
test_filename = None
task = None
idx = "id"
targets = ["service_a", "service_b"]
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
