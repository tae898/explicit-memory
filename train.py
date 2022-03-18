"""Import necessary stuff and start training."""
import logging

logger = logging.getLogger()
logger.disabled = True

import argparse
import os
import shutil
from datetime import datetime
from pprint import pformat

from memory.trainer import Trainer
from memory.utils import read_yaml, seed_everything


def main(**kwargs):
    trainer = Trainer(**kwargs)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train RL with arguments.")
    parser.add_argument(
        "--config", type=str, default="./train.yaml", help="path to the config file."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="training-results",
        help="log and ckpt save dir",
    )
    args = parser.parse_args()
    config = read_yaml(args.config)

    model_summary = "-".join(
        [
            str(value)
            if not isinstance(value, dict)
            else "-".join(str(foo) for foo in list(value.values()))
            for value in config["strategies"].values()
        ]
    )

    current_time = "_".join(str(datetime.now()).split())

    save_dir = os.path.join(args.save_dir, model_summary, current_time)
    config_copy_dst = os.path.join(save_dir, "train.yaml")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(
        args.config,
        config_copy_dst,
    )

    print(f"\nArguments\n---------\n{pformat(config,indent=4, width=1)}\n")

    main(
        **config,
        save_dir=save_dir,
    )
