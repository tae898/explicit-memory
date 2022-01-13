"""utility functions"""
import csv
import json
import logging
import os
import pickle
import random
import numpy as np
import torch
import yaml

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def seed_everything(seed: int):
    """Seed every randomness to seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_json(fname: str) -> dict:
    """Read json"""
    logging.debug(f"reading json {fname} ...")
    with open(fname, "r") as stream:
        return json.load(stream)


def write_json(content: dict, fname: str) -> None:
    """Write json"""
    logging.debug(f"writing json {fname} ...")
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> dict:
    """Read yaml."""
    logging.debug(f"reading yaml {fname} ...")
    with open(fname, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content: dict, fname: str) -> None:
    """Read yaml."""
    logging.debug(f"writing yaml {fname} ...")
    with open(fname, "w") as stream:
        return yaml.dump(content, stream, indent=4, sort_keys=False)


def read_pickle(fname: str):
    """Read pickle"""
    logging.debug(f"writing pickle {fname} ...")
    with open(fname, "rb") as stream:
        foo = pickle.load(stream)
    return foo


def write_csv(content: list, fname: str) -> None:
    with open(fname, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(content)


def read_data(data_path: str) -> dict:
    """Read train, val, test spilts.

    Args
    ----
    data_path: path to data.

    Returns
    -------
    data: {'train': list of training obs,
           'val': list of val obs,
           'test': list of test obs}
    """
    logging.debug(f"reading data from {data_path} ...")
    data = read_json(data_path)
    logging.info(f"Succesfully read data {data_path}")

    return data


def load_questions(path: str) -> dict:
    """Load premade questions.

    Args
    ----
    path: path to the question json file.

    """
    logging.debug(f"loading questions from {path}...")
    questions = read_json(path)
    logging.info(f"questions loaded from {path}!")

    return questions


# def is_clean(self, edge: dict) -> None:
#     """See if conceptnet query is clean or not.

#     I really tried to accept as much noise as possible, since I didn't want
#     to manually clean the data, but some are really dirty. I gotta clean them.

#     Args
#     ----
#     edge: edge from the ConceptNet query output.

#     """
#     logging.debug(f"Checking if {edge} is clean or not ...")
#     if edge["end"]["@id"].split("/")[-1] in self.dirty_tails:
#         return False
#     else:
#         return True

# def read_dirty_tails(self, path: str = "./data/dirty_tails") -> None:
#     """Read dirty tails.

#     Args
#     ----
#     path: The path to the dirty tail list.

#     """
#     logging.debug(f"Reading {path} ...")
#     with open(path, "r") as stream:
#         self.dirty_tails = stream.readlines()
#     self.dirty_tails = [line.strip() for line in self.dirty_tails]
#     logging.info(
#         f"Reading {path} complete! There are {len(self.dirty_tails)} dirty tails!"
#     )
