from dataclasses import dataclass
from functools import partial

import numpy as np
from sklearn import metrics as skmetrics

from .enums import ProblemType
import copy


@dataclass
class Metrics:
    problem_type: ProblemType

    def __post_init__(self):
        if self.problem_type == ProblemType.binary_classification:
            self.valid_metrics = {
                "auc": skmetrics.roc_auc_score,
                "logloss": skmetrics.log_loss,
                "f1": skmetrics.f1_score,
                "accuracy": skmetrics.accuracy_score,
                "precision": skmetrics.precision_score,
                "recall": skmetrics.recall_score,
            }
        elif self.problem_type == ProblemType.multi_class_classification:
            self.valid_metrics = {
                "logloss": skmetrics.log_loss,
                "accuracy": skmetrics.accuracy_score,
                "mlogloss": skmetrics.log_loss,
            }
        elif self.problem_type in (ProblemType.single_column_regression, ProblemType.multi_column_regression):
            self.valid_metrics = {
                "r2": skmetrics.r2_score,
                "mse": skmetrics.mean_squared_error,
                "mae": skmetrics.mean_absolute_error,
                "rmse": partial(skmetrics.mean_squared_error, squared=False),
                "rmsle": partial(skmetrics.mean_squared_log_error, squared=False),
            }
        elif self.problem_type == ProblemType.multi_label_classification:
            self.valid_metrics = {
                "logloss": skmetrics.log_loss,
            }
        else:
            raise Exception("Invalid problem type")

    def calculate(self, y_true, y_pred):
        metrics = {}
        for metric_name, metric_func in self.valid_metrics.items():
            if self.problem_type == ProblemType.binary_classification:
                if metric_name == "auc":
                    metrics[metric_name] = metric_func(y_true, y_pred[:, 1])
                elif metric_name == "logloss":
                    metrics[metric_name] = metric_func(y_true, y_pred)
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred[:, 1] >= 0.5)
            elif self.problem_type == ProblemType.multi_class_classification:
                if metric_name == "accuracy":
                    metrics[metric_name] = metric_func(y_true, np.argmax(y_pred, axis=1))
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred)
            else:
                if metric_name == "rmsle":
                    temp_pred = copy.deepcopy(y_pred)
                    temp_pred[temp_pred < 0] = 0
                    metrics[metric_name] = metric_func(y_true, temp_pred)
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred)
        return metrics
