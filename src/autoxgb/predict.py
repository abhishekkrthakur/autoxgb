import json
import os
from dataclasses import dataclass
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import create_model

from .enums import ProblemType
from .utils import fetch_xgb_model_params, reduce_memory_usage


xgb.set_config(verbosity=0)


@dataclass
class AutoXGBPredict:
    model_path: str

    def __post_init__(self):
        self.model_config = joblib.load(os.path.join(self.model_path, "axgb.config"))
        self.target_encoder = joblib.load(os.path.join(self.model_path, "axgb.target_encoder"))
        self.categorical_encoders = joblib.load(os.path.join(self.model_path, "axgb.categorical_encoders"))
        self.models = []
        for fold in range(self.model_config.num_folds):
            model_ = joblib.load(os.path.join(self.model_path, f"axgb_model.{fold}"))
            self.models.append(model_)

        _, self.use_predict_proba, _, _ = fetch_xgb_model_params(self.model_config)

    # TODO: improve schema with optional null values
    def get_prediction_schema(self):
        cat_features = self.model_config.categorical_features
        schema = {"PredictSchema": {}}
        for cf in cat_features:
            schema["PredictSchema"][cf] = "str"

        for feat in self.model_config.features:
            if feat not in cat_features:
                schema["PredictSchema"][feat] = 10.0
        return create_model("PredictSchema", **schema["PredictSchema"])

    def _predict_df(self, df):
        categorical_features = self.model_config.categorical_features
        final_preds = []
        test_ids = df[self.model_config.idx].values
        for fold in range(self.model_config.num_folds):
            fold_test = df.copy(deep=True)
            if len(categorical_features) > 0:
                categorical_encoder = self.categorical_encoders[fold]
                fold_test[categorical_features] = categorical_encoder.transform(fold_test[categorical_features].values)

            test_features = fold_test[self.model_config.features]

            for col in test_features.columns:
                if test_features[col].dtype == "object":
                    test_features[col] = test_features[col].astype(np.int64)

            if self.model_config.problem_type in (
                ProblemType.multi_column_regression,
                ProblemType.multi_label_classification,
            ):
                test_preds_mll = []
                for midx in range(len(self.models[fold])):
                    if self.model_config.problem_type == ProblemType.multi_column_regression:
                        test_pred_temp = self.models[fold][midx].predict(test_features)
                    else:
                        test_pred_temp = self.models[fold][midx].predict_proba(test_features)[:, 1]
                    test_preds_mll.append(test_pred_temp)

                test_preds = np.column_stack(test_preds_mll)
            else:
                if self.use_predict_proba:
                    test_preds = self.models[fold].predict_proba(test_features)
                else:
                    test_preds = self.models[fold].predict(test_features)

            final_preds.append(test_preds)

        final_preds = np.mean(final_preds, axis=0)
        if self.target_encoder is None:
            final_preds = pd.DataFrame(final_preds, columns=self.model_config.target_cols)
        else:
            final_preds = pd.DataFrame(final_preds, columns=list(self.target_encoder.classes_))
        final_preds.insert(loc=0, column=self.model_config.idx, value=test_ids)
        return final_preds

    def predict_single(self, sample: Dict[str, Union[str, int, float]] = None, fast_predict: bool = True):
        sample = json.loads(sample)
        sample_df = pd.DataFrame.from_dict(sample, orient="index").T
        sample_df[self.model_config.idx] = 0
        preds = self._predict_df(sample_df)
        preds = preds.to_dict(orient="records")[0]
        return preds

    def predict_file(self, test_filename: str, out_filename: str):
        test_df = pd.read_csv(test_filename)
        test_df = reduce_memory_usage(test_df)
        final_preds = self._predict_df(test_df)
        final_preds.to_csv(out_filename, index=False)
