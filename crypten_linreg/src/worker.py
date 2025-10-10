import logging
import os
import sys
import argparse
from importlib import import_module

from config import config
from tasks import REGISTRY


logging.basicConfig(
    level=config.LOGLEVEL,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(prog="worker")

    subparsers = parser.add_subparsers(dest="task", required=True)

    for name in REGISTRY.keys():
        subparsers.add_parser(name)

    dotted = subparsers.add_parser("call")
    dotted.add_argument("target", help="dotted path, напр. tasks.collective:reduce")

    args = parser.parse_args()

    if args.task == "call":
        module_path, func_name = args.target.split(":")
        fn = getattr(import_module(module_path), func_name)
    else:
        fn = REGISTRY[args.task]

    fn()

if __name__ == "__main__":
    main()
