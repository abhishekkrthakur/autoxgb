import copy
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
from .params import get_params


def save_valid_predictions(final_valid_predictions, model_config, target_encoder, output_file_name):
    final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    if target_encoder is None:
        final_valid_predictions.columns = [model_config.id_column] + model_config.target_cols
    else:
        final_valid_predictions.columns = [model_config.id_column] + list(target_encoder.classes_)

    final_valid_predictions.to_csv(
        os.path.join(model_config.output_dir, output_file_name),
        index=False,
    )


def save_test_predictions(final_test_predictions, model_config, target_encoder, test_ids, output_file_name):
    final_test_predictions = np.mean(final_test_predictions, axis=0)
    if target_encoder is None:
        final_test_predictions = pd.DataFrame(final_test_predictions, columns=model_config.target_cols)
    else:
        final_test_predictions = pd.DataFrame(final_test_predictions, columns=list(target_encoder.classes_))
    final_test_predictions.insert(loc=0, column=model_config.id_column, value=test_ids)
    final_test_predictions.to_csv(
        os.path.join(model_config.output_dir, output_file_name),
        index=False,
    )


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
    elif model_config.problem_type == ProblemType.multi_label_classification:
        metric = metrics.log_loss
        xgb_model = xgb.XGBClassifier
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "logloss"
    elif model_config.problem_type == ProblemType.single_column_regression:
        metric = partial(metrics.mean_squared_error, squared=False)
        xgb_model = xgb.XGBRegressor
        use_predict_proba = False
        direction = "minimize"
        eval_metric = "rmse"
    elif model_config.problem_type == ProblemType.multi_column_regression:
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
    metric,
    use_predict_proba,
    eval_metric,
    model_config,
):
    params = get_params(trial, model_config)
    early_stopping_rounds = params["early_stopping_rounds"]
    del params["early_stopping_rounds"]

    scores = []

    for fold in range(model_config.num_folds):
        train_feather = pd.read_feather(os.path.join(model_config.output_dir, f"train_fold_{fold}.feather"))
        valid_feather = pd.read_feather(os.path.join(model_config.output_dir, f"valid_fold_{fold}.feather"))
        xtrain = train_feather[model_config.features]
        xvalid = valid_feather[model_config.features]

        ytrain = train_feather[model_config.target_cols].values
        yvalid = valid_feather[model_config.target_cols].values

        # train model
        model = xgb_model(
            random_state=model_config.seed,
            eval_metric=eval_metric,
            use_label_encoder=False,
            **params,
        )

        if model_config.problem_type in (ProblemType.multi_column_regression, ProblemType.multi_label_classification):
            ypred = []
            models = [model] * len(model_config.target_cols)
            for idx, _m in enumerate(models):
                _m.fit(
                    xtrain,
                    ytrain[:, idx],
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=[(xvalid, yvalid[:, idx])],
                    verbose=False,
                )
                if model_config.problem_type == ProblemType.multi_column_regression:
                    ypred_temp = _m.predict(xvalid)
                else:
                    ypred_temp = _m.predict_proba(xvalid)[:, 1]
                ypred.append(ypred_temp)
            ypred = np.column_stack(ypred)

        else:
            model.fit(
                xtrain,
                ytrain,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=[(xvalid, yvalid)],
                verbose=False,
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
        metric=metric,
        use_predict_proba=use_predict_proba,
        eval_metric=eval_metric,
        model_config=model_config,
    )
    db_path = os.path.join(model_config.output_dir, "params.db")
    study = optuna.create_study(
        direction=direction,
        study_name="autoxgb",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )
    study.optimize(optimize_func, n_trials=model_config.num_trials, timeout=model_config.time_limit)
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

        model = xgb_model(
            random_state=model_config.seed,
            eval_metric=eval_metric,
            use_label_encoder=False,
            **best_params,
        )

        if model_config.problem_type in (ProblemType.multi_column_regression, ProblemType.multi_label_classification):
            ypred = []
            test_pred = []
            trained_models = []
            for idx in range(len(model_config.target_cols)):
                _m = copy.deepcopy(model)
                _m.fit(
                    xtrain,
                    ytrain[:, idx],
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=[(xvalid, yvalid[:, idx])],
                    verbose=False,
                )
                trained_models.append(_m)
                if model_config.problem_type == ProblemType.multi_column_regression:
                    ypred_temp = _m.predict(xvalid)
                    if model_config.test_filename is not None:
                        test_pred_temp = _m.predict(xtest)
                else:
                    ypred_temp = _m.predict_proba(xvalid)[:, 1]
                    test_pred_temp = _m.predict_proba(xtest)[:, 1]

                ypred.append(ypred_temp)
                test_pred.append(test_pred_temp)

            ypred = np.column_stack(ypred)
            test_pred = np.column_stack(test_pred)
            joblib.dump(
                trained_models,
                os.path.join(
                    model_config.output_dir,
                    f"axgb_model.{fold}",
                ),
            )

        else:
            model.fit(
                xtrain,
                ytrain,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=[(xvalid, yvalid)],
                verbose=False,
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

    save_valid_predictions(final_valid_predictions, model_config, target_encoder, "oof_predictions.csv")

    if model_config.test_filename is not None:
        save_test_predictions(final_test_predictions, model_config, target_encoder, test_ids, "test_predictions.csv")
    else:
        logger.info("No test data supplied. Only OOF predictions were generated")
