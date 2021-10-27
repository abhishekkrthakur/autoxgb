from typing import List, Optional

from pydantic import BaseModel

from .enums import ProblemType


class ModelConfig(BaseModel):
    train_filename: str
    test_filename: Optional[str] = None
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
