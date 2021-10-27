import os
from functools import partial

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn import metrics

from .enums import ProblemType


def fetch_xgb_model_params(model_config):
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
    elif model_config.problem_type == ProblemType.single_column_regression:
        metric = partial(metrics.mean_squared_error, squared=False)
        xgb_model = xgb.XGBRegressor
        use_predict_proba = False
        direction = "minimize"
        eval_metric = "rmse"
    else:
        raise NotImplementedError

    return xgb_model, use_predict_proba, eval_metric, metric, direction


def optimize(
    trial,
    xgb_model,
    num_folds,
    features,
    targets,
    metric,
    config_dir,
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
        train_feather = pd.read_feather(os.path.join(config_dir, f"train_fold_{fold}.feather"))
        valid_feather = pd.read_feather(os.path.join(config_dir, f"valid_fold_{fold}.feather"))
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
    xgb_model, use_predict_proba, eval_metric, metric, direction = fetch_xgb_model_params(model_config)

    optimize_func = partial(
        optimize,
        xgb_model=xgb_model,
        num_folds=model_config.num_folds,
        features=model_config.features,
        targets=model_config.target_cols,
        metric=metric,
        config_dir=model_config.output_dir,
        use_predict_proba=use_predict_proba,
        eval_metric=eval_metric,
    )
    db_path = os.path.join(model_config.output_dir, "params.db")
    study = optuna.create_study(
        direction=direction,
        study_name="autoxgb",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )
    study.optimize(optimize_func, n_trials=1)
    return study.best_params


def predict_model(model_config, best_params):

    early_stopping_rounds = best_params["early_stopping_rounds"]
    del best_params["early_stopping_rounds"]

    xgb_model, use_predict_proba, eval_metric, metric, _ = fetch_xgb_model_params(model_config)
    scores = []

    final_test_predictions = []
    final_valid_predictions = {}

    target_encoder = joblib.load(f"{model_config.output_dir}/axgb.target_encoder")

    for fold in range(model_config.num_folds):
        train_feather = pd.read_feather(os.path.join(model_config.output_dir, f"train_fold_{fold}.feather"))
        valid_feather = pd.read_feather(os.path.join(model_config.output_dir, f"valid_fold_{fold}.feather"))

        xtrain = train_feather[model_config.features]
        xvalid = valid_feather[model_config.features]

        valid_ids = valid_feather[model_config.id_column].values

        if model_config.test_filename is not None:
            test_feather = pd.read_feather(os.path.join(model_config.output_dir, f"test_fold_{fold}.feather"))
            xtest = test_feather[model_config.features]
            test_ids = test_feather[model_config.id_column].values

        ytrain = train_feather[model_config.target_cols].values
        yvalid = valid_feather[model_config.target_cols].values

        # train model
        model = xgb_model(
            random_state=42,
            tree_method="gpu_hist",
            gpu_id=1,
            predictor="gpu_predictor",
            n_estimators=7000,
            eval_metric=eval_metric,
            use_label_encoder=False,
            **best_params,
        )
        model.fit(
            xtrain,
            ytrain,
            early_stopping_rounds=early_stopping_rounds,
            eval_set=[(xvalid, yvalid)],
            verbose=1000,
        )
        joblib.dump(
            model,
            os.path.join(
                model_config.output_dir,
                f"axgb_model.{fold}",
            ),
        )

        if use_predict_proba:
            ypred = model.predict_proba(xvalid)
            if model_config.test_filename is not None:
                test_pred = model.predict_proba(xtest)
        else:
            ypred = model.predict(xvalid)
            if model_config.test_filename is not None:
                test_pred = model.predict(xtest)

        final_valid_predictions.update(dict(zip(valid_ids, ypred)))
        if model_config.test_filename is not None:
            final_test_predictions.append(test_pred)

        # calculate metric
        metric_value = metric(yvalid, ypred)
        scores.append(metric_value)

    final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    if target_encoder is None:
        final_valid_predictions.columns = [model_config.id_column] + model_config.target_cols
    else:
        final_valid_predictions.columns = [model_config.id_column] + list(target_encoder.classes_)

    final_valid_predictions.to_csv(
        os.path.join(model_config.output_dir, "oof_predictions.csv"),
        index=False,
    )

    if model_config.test_filename is not None:
        final_test_predictions = np.mean(final_test_predictions, axis=0)
        if target_encoder is None:
            final_test_predictions = pd.DataFrame(final_test_predictions, columns=model_config.target_cols)
        else:
            final_test_predictions = pd.DataFrame(final_test_predictions, columns=list(target_encoder.classes_))
        final_test_predictions.insert(loc=0, column=model_config.id_column, value=test_ids)
        final_test_predictions.to_csv(
            os.path.join(model_config.output_dir, "test_predictions.csv"),
            index=False,
        )
    else:
        logger.info("No test data supplied. Only OOF predictions were generated")
