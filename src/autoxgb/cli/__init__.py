from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        raise NotImplementedError()
