"""Lazy command loading keeps --help usable without a graphics context."""

import argparse
import importlib
import sys

COMMANDS = {
    "view": ("panthera_lab.apps.viewer", "Native MuJoCo viewer and keyboard chassis"),
    "doctor": (
        "panthera_lab.apps.doctor",
        "Check versions, assets, weights and optional rendering",
    ),
    "eval": ("panthera_lab.apps.evaluate", "Evaluate an installed task with independent metrics"),
    "train": ("panthera_lab.learning.train", "Train/resume nine-action residual PPO"),
    "record": ("panthera_lab.apps.record", "Record RGB-D and motor packets for image policies"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="panthera", description="Panthera wheeled-arm experiment framework"
    )
    parser.add_argument(
        "command", choices=COMMANDS, help=" | ".join(f"{k}: {v[1]}" for k, v in COMMANDS.items())
    )
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in ("-h", "--help"):
        parser.print_help()
        return
    args = parser.parse_args(tokens[:1])
    importlib.import_module(COMMANDS[args.command][0]).main(tokens[1:])
