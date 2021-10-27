from argparse import ArgumentParser

from ..autoxgb import AutoXGB
from . import BaseCommand


def train_autoxgb_command_factory(args):
    return TrainAutoXGBCommand(
        args.train_filename,
        args.id_column,
        args.target,
        args.problem_type,
        args.output_dir,
        args.features,
        args.num_folds,
        args.use_gpu,
        args.seed,
        args.test_filename,
    )


class TrainAutoXGBCommand(BaseCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("train", help="Train a new model using AutoXGB")
        _parser.add_argument("--train_filename", help="Path to training file", required=True, type=str)
        _parser.add_argument("--test_filename", help="Path to test file", required=False, type=str, default=None)
        _parser.add_argument("--output_dir", help="Path to output directory", required=True, type=str)
        _parser.add_argument("--problem_type", help="Problem type", required=False, type=str, default="classification")
        _parser.add_argument("--id_column", help="ID column", required=False, type=str, default="id")
        _parser.add_argument("--target", help="Target column", required=False, type=str, default="target")
        _parser.add_argument("--num_folds", help="Number of folds to use", required=False, type=int, default=5)
        _parser.add_argument(
            "--features", help="Features to use, separated by ';'", required=False, type=str, default=None
        )
        _parser.add_argument(
            "--use_gpu", help="Whether to use GPU for training", required=False, type=bool, default=False
        )
        _parser.add_argument("--seed", help="Random seed", required=False, type=int, default=42)
        _parser.set_defaults(func=train_autoxgb_command_factory)

    def __init__(
        self,
        train_filename,
        id_column,
        target,
        problem_type,
        output_dir,
        features,
        num_folds,
        use_gpu,
        seed,
        test_filename,
    ):
        self.train_filename = train_filename
        self.id_column = id_column
        self.target = target.split(";")
        self.problem_type = problem_type
        self.output_dir = output_dir
        self.features = features.split(";") if features else None
        self.num_folds = num_folds
        self.use_gpu = use_gpu
        self.seed = seed
        self.test_filename = test_filename

    def execute(self):
        axgb = AutoXGB(
            train_filename=self.train_filename,
            id_col=self.id_column,
            target_cols=self.target,
            problem_type=self.problem_type,
            output_dir=self.output_dir,
            features=self.features,
            num_folds=self.num_folds,
            use_gpu=self.use_gpu,
            seed=self.seed,
            test_filename=self.test_filename,
        )
        axgb.train()
