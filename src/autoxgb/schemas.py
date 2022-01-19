from typing import List, Optional, Callable

from pydantic import BaseModel

from .enums import ProblemType
import pandas as pd


class ModelConfig(BaseModel):
    train_filename: str
    test_filename: Optional[str] = None
    idx: str
    targets: List[str]
    problem_type: ProblemType
    output: str
    features: List[str]
    num_folds: int
    use_gpu: bool
    seed: int
    categorical_features: List[str]
    num_trials: int
    time_limit: Optional[int] = None
    fast: bool
    data_aug_func: Optional[Callable[[pd.DataFrame, 'ModelConfig', int], pd.DataFrame]] = None
    
