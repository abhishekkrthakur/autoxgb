import os
from dataclasses import dataclass
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .utils import fetch_xgb_model_params


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

        _, self.use_predict_proba, _, _, _ = fetch_xgb_model_params(self.model_config)

    def _predict_df(self, df):
        categorical_features = self.model_config.categorical_features
        final_preds = []
        for fold in range(self.model_config.num_folds):
            fold_test = df.copy(deep=True)
            if len(categorical_features) > 0:
                categorical_encoder = self.categorical_encoders[fold]
                fold_test[categorical_features] = categorical_encoder.transform(fold_test[categorical_features].values)
            test_features = fold_test[self.model_config.features]
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
        return final_preds

    def predict_single(self, sample: Dict[str, Union[str, int, float]] = None, fast_predict: bool = True):
        sample_df = pd.DataFrame(sample)
        _ = self._predict_df(sample_df)

    def predict_file(self, test_filename: str, output_dir: str):
        test_df = pd.read_csv(test_filename)
        final_preds = self._predict_df(test_df)
        final_preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
