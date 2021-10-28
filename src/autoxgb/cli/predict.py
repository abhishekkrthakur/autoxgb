from argparse import ArgumentParser

from ..predict import AutoXGBPredict
from . import BaseCommand


def predict_autoxgb_command_factory(args):
    return PredictAutoXGBCommand(args.model_path, args.test_filename, args.out_filename)


class PredictAutoXGBCommand(BaseCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("predict", help="Make predictions on any CSV file")
        _parser.add_argument("--model_path", help="Path to model", required=True, type=str)
        _parser.add_argument("--test_filename", help="Path to test file", required=False, type=str, default=None)
        _parser.add_argument("--out_filename", help="Path to output file", required=True, type=str)
        _parser.set_defaults(func=predict_autoxgb_command_factory)

    def __init__(self, model_path, test_filename, out_filename):
        self.model_path = model_path
        self.test_filename = test_filename
        self.out_filename = out_filename

    def execute(self):
        axgb = AutoXGBPredict(model_path=self.model_path)
        axgb.predict_file(test_filename=self.test_filename, out_filename=self.out_filename)
