import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser

from loguru import logger


logger.configure(
    handlers=[
        dict(
            sink=sys.stderr,
            format="<level>axgb: {time:YYYY-MM-DD HH:mm:ss} {level:<7} </level> <cyan>| {message}</cyan>",
        )
    ]
)


class BaseCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        raise NotImplementedError()
