import joblib
import pandas as pd
import optuna
from functools import partial
import xgboost as xgb
from .enums import ProblemType
from sklearn import metrics
import os


def optimize(
    trial,
    xgb_model,
    num_folds,
    features,
    targets,
    metric,
    config_dir,
    model_identifier,
    use_predict_proba,
    eval_metric,
):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)
    early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 100, 500)

    scores = []

    for fold in range(num_folds):
        train_feather = pd.read_feather(f"{config_dir}/{model_identifier}/train_fold_{fold}.feather")
        valid_feather = pd.read_feather(f"{config_dir}/{model_identifier}/valid_fold_{fold}.feather")
        xtrain = train_feather[features]
        xvalid = valid_feather[features]

        ytrain = train_feather[targets].values
        yvalid = valid_feather[targets].values

        # train model
        model = xgb_model(
            random_state=42,
            tree_method="gpu_hist",
            gpu_id=1,
            predictor="gpu_predictor",
            n_estimators=7000,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            max_depth=max_depth,
            eval_metric=eval_metric,
            use_label_encoder=False,
        )
        model.fit(
            xtrain,
            ytrain,
            early_stopping_rounds=early_stopping_rounds,
            eval_set=[(xvalid, yvalid)],
            verbose=1000,
        )

        if use_predict_proba:
            ypred = model.predict_proba(xvalid)
        else:
            ypred = model.predict(xvalid)

        # calculate metric
        metric_value = metric(yvalid, ypred)
        scores.append(metric_value)

    return sum(scores) / len(scores)


def train_model(model_config):
    if model_config.problem_type == ProblemType.binary_classification:
        metric = metrics.log_loss
        xgb_model = xgb.XGBClassifier
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "logloss"
    elif model_config.problem_type == ProblemType.multi_class_classification:
        metric = metrics.log_loss
        xgb_model = xgb.XGBClassifier
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "mlogloss"
    else:
        raise NotImplementedError

    optimize_func = partial(
        optimize,
        xgb_model=xgb_model,
        num_folds=model_config.num_folds,
        features=model_config.features,
        targets=model_config.target_cols,
        metric=metric,
        config_dir=model_config.output_dir,
        model_identifier=model_config.model_identifier,
        use_predict_proba=use_predict_proba,
        eval_metric=eval_metric,
    )
    db_path = os.path.join(model_config.output_dir, f"{model_config.model_identifier}.db")
    study = optuna.create_study(
        direction=direction,
        study_name=model_config.model_identifier,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )
    study.optimize(optimize_func, n_trials=5)
