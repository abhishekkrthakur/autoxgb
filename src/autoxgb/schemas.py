from pydantic import BaseModel
from typing import List

from .enums import ProblemType


class ModelConfig(BaseModel):
    train_filename: str
    id_column: str
    target_cols: List[str]
    problem_type: ProblemType
    output_dir: str
    features: List[str]
    num_folds: int
    use_gpu: bool
    seed: int
    version: str
    categorical_features: List[str]
    model_identifier: str
