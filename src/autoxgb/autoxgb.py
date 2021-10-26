import os
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from . import __version__

from .train import train_model
from .schemas import ModelConfig
from .enums import ProblemType
import numpy as np


@dataclass
class AutoXGB:
    train_filename: str
    output_dir: str
    name: str
    problem_type: Optional[str] = "binary_classification"
    id_col: Optional[str] = "id"
    target_cols: Optional[List[str]] = None
    features: Optional[List[str]] = None
    use_gpu: Optional[bool] = False
    num_folds: Optional[int] = 5
    seed: Optional[int] = 42

    def __post_init__(self):
        self._version = __version__
        if os.path.exists(self.output_dir):
            logger.warning(f"Output directory {self.output_dir} already exists. Will overwrite existing files.")
        os.makedirs(os.path.join(self.output_dir, self.name), exist_ok=True)

        if self.target_cols is None:
            logger.warning("No target columns specified. Will default to `target`.")
            self.target_cols = ["target"]

        if self.id_col is None:
            logger.warning("No id column specified. Will default to `id`.")
            self.id_col = "id"

    def _create_folds(self, train_df, problem_type):
        logger.info("Creating folds")
        train_df["kfold"] = -1
        if problem_type in (ProblemType.binary_classification, ProblemType.multi_class_classification):
            y = train_df[self.target_cols].values
            kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            for fold, (_, valid_indicies) in enumerate(kf.split(X=train_df, y=y)):
                train_df.loc[valid_indicies, "kfold"] = fold
        else:
            raise Exception("Problem type not supported")
        return train_df

    def _determine_problem_type(self, train_df):
        if self.problem_type == "classification":
            if len(self.target_cols) == 1:
                if len(np.unique(train_df[self.target_cols].values)) == 2:
                    problem_type = ProblemType.binary_classification
                else:
                    problem_type = ProblemType.multi_class_classification
            else:
                problem_type = ProblemType.multi_label_classification
                raise NotImplementedError("Multi-label classification not supported yet.")

        elif self.problem_type == "regression":
            if len(self.target_cols) == 1:
                problem_type = ProblemType.single_column_regression
                raise NotImplementedError("Single-column regression not supported yet.")
            else:
                problem_type = ProblemType.multi_column_regression
                raise NotImplementedError("Multi-column regression not supported yet.")

        else:
            raise Exception("Problem type not understood")

        return problem_type

    def _process_data(self):
        logger.info("Reading training data")
        train_df = pd.read_csv(self.train_filename)
        problem_type = self._determine_problem_type(train_df)

        # create folds
        train_df = self._create_folds(train_df, problem_type)
        ignore_columns = [self.id_col, "kfold"] + self.target_cols

        if self.features is None:
            self.features = list(train_df.columns)
            self.features = [x for x in self.features if x not in ignore_columns]

        # encode target(s)
        if problem_type in [ProblemType.binary_classification, ProblemType.multi_class_classification]:
            logger.info("Encoding target(s)")
            target_encoder = LabelEncoder()
            target_encoder.fit(
                train_df[self.target_cols].values.reshape(
                    -1,
                )
            )
            train_df.loc[:, self.target_cols] = target_encoder.transform(
                train_df[self.target_cols].values.reshape(
                    -1,
                )
            )
        else:
            target_encoder = None

        # find categorical features
        categorical_features = []
        for col in self.features:
            if train_df[col].dtype == "object":
                categorical_features.append(col)

        logger.info(f"Found {len(categorical_features)} categorical features.")

        if len(categorical_features) > 0:
            logger.info("Encoding categorical features")
        categorical_encoders = {}
        for fold in range(self.num_folds):
            fold_train = train_df[train_df.kfold != fold].reset_index(drop=True)
            fold_valid = train_df[train_df.kfold == fold].reset_index(drop=True)
            if len(categorical_features) > 0:
                ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999)
                fold_train[categorical_features] = ord_encoder.fit_transform(fold_train[categorical_features].values)
                fold_valid[categorical_features] = ord_encoder.transform(fold_valid[categorical_features].values)
                categorical_encoders[fold] = ord_encoder
            fold_train.to_feather(f"{self.output_dir}/{self.name}/train_fold_{fold}.feather")
            fold_valid.to_feather(f"{self.output_dir}/{self.name}/valid_fold_{fold}.feather")

        # save config
        model_config = {}
        model_config["id_column"] = self.id_col
        model_config["features"] = self.features
        model_config["categorical_features"] = categorical_features
        model_config["train_filename"] = self.train_filename
        model_config["output_dir"] = self.output_dir
        model_config["problem_type"] = problem_type
        model_config["id_col"] = self.id_col
        model_config["target_cols"] = self.target_cols
        model_config["use_gpu"] = self.use_gpu
        model_config["num_folds"] = self.num_folds
        model_config["seed"] = self.seed
        model_config["model_identifier"] = self.name
        model_config["version"] = self._version

        self.model_config = ModelConfig(**model_config)
        logger.info(f"Model config: {self.model_config}")
        logger.info("Saving model config")
        joblib.dump(self.model_config, f"{self.output_dir}/axgb.config")

        # save encoders
        logger.info("Saving encoders")
        joblib.dump(categorical_encoders, f"{self.output_dir}/axgb.categorical_encoders")
        joblib.dump(target_encoder, f"{self.output_dir}/axgb.target_encoder")

    def train(self):
        self._process_data()
        train_model(self.model_config)
        logger.info("Training complete")

    def predict(self):
        pass
