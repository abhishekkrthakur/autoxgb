from argparse import ArgumentParser

from ..autoxgb import AutoXGB
from ..enums import TaskType
from . import BaseCommand


def train_autoxgb_command_factory(args):
    return TrainAutoXGBCommand(
        args.train_filename,
        args.idx,
        args.targets,
        args.task,
        args.output,
        args.features,
        args.num_folds,
        args.use_gpu,
        args.seed,
        args.test_filename,
        args.time_limit,
        args.fast,
    )


class TrainAutoXGBCommand(BaseCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("train", help="Train a new model using AutoXGB")
        _parser.add_argument(
            "--train_filename",
            help="Path to training file",
            required=True,
            type=str,
        )
        _parser.add_argument(
            "--test_filename",
            help="Path to test file",
            required=False,
            type=str,
            default=None,
        )
        _parser.add_argument(
            "--output",
            help="Path to output directory",
            required=True,
            type=str,
        )
        _parser.add_argument(
            "--task",
            help="User defined task type",
            required=False,
            type=str,
            default=None,
            choices=TaskType.list_str(),
        )
        _parser.add_argument(
            "--idx",
            help="ID column",
            required=False,
            type=str,
            default="id",
        )
        _parser.add_argument(
            "--targets",
            help="Target column(s). If there are multiple targets, separate by ';'",
            required=False,
            type=str,
            default="target",
        )
        _parser.add_argument(
            "--num_folds",
            help="Number of folds to use",
            required=False,
            type=int,
            default=5,
        )
        _parser.add_argument(
            "--features",
            help="Features to use, separated by ';'",
            required=False,
            type=str,
            default=None,
        )
        _parser.add_argument(
            "--use_gpu",
            help="Whether to use GPU for training",
            action="store_true",
            required=False,
        )
        _parser.add_argument(
            "--fast",
            help="Whether to use fast mode for tuning params. Only one fold will be used if fast mode is set",
            action="store_true",
            required=False,
        )
        _parser.add_argument(
            "--seed",
            help="Random seed",
            required=False,
            type=int,
            default=42,
        )
        _parser.add_argument(
            "--time_limit",
            help="Time limit for optimization",
            required=False,
            type=int,
            default=None,
        )

        _parser.set_defaults(func=train_autoxgb_command_factory)

    def __init__(
        self,
        train_filename,
        idx,
        targets,
        task,
        output,
        features,
        num_folds,
        use_gpu,
        seed,
        test_filename,
        time_limit,
        fast,
    ):
        self.train_filename = train_filename
        self.idx = idx
        self.targets = targets.split(";")
        self.task = task
        self.output = output
        self.features = features.split(";") if features else None
        self.num_folds = num_folds
        self.use_gpu = use_gpu
        self.seed = seed
        self.test_filename = test_filename
        self.time_limit = time_limit
        self.fast = fast

    def execute(self):
        axgb = AutoXGB(
            train_filename=self.train_filename,
            idx=self.idx,
            targets=self.targets,
            task=self.task,
            output=self.output,
            features=self.features,
            num_folds=self.num_folds,
            use_gpu=self.use_gpu,
            seed=self.seed,
            test_filename=self.test_filename,
            time_limit=self.time_limit,
            fast=self.fast,
        )
        axgb.train()
