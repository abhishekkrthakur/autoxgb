import os
from argparse import ArgumentParser

import uvicorn

from . import BaseCommand


def serve_autoxgb_command_factory(args):
    return ServeAutoXGBCommand(args.model_path, args.port, args.host, args.workers, args.debug)


class ServeAutoXGBCommand(BaseCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("serve", help="Serve AutoXGB API")
        _parser.add_argument("--model_path", help="Path to model", required=True, type=str)
        _parser.add_argument("--port", help="Port to serve on", default=9999, type=int, required=False)
        _parser.add_argument("--host", help="Host to serve on", default="127.0.0.1", type=str, required=False)
        _parser.add_argument("--workers", help="Number of workers", default=1, type=int, required=False)
        _parser.add_argument("--debug", help="Debug mode", action="store_true", required=False)
        _parser.set_defaults(func=serve_autoxgb_command_factory)

    def __init__(self, model_path, port, host, workers, debug):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.workers = workers
        self.debug = debug

    def execute(self):
        os.environ["AUTOXGB_MODEL_PATH"] = self.model_path
        # run app using uvicorn
        uvicorn.run(
            "autoxgb.api:app",
            host=self.host,
            port=self.port,
            debug=self.debug,
            workers=self.workers,
        )
