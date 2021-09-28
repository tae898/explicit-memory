"""utility functions"""
import json
import pickle


def read_json(fname: str):
    """Read json"""
    with open(fname, 'r') as stream:
        return json.load(stream)


def write_json(content, fname: str):
    """Write json"""
    with open(fname, 'w') as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_pickle(pickle_path: str):
    """Read pickle"""
    with open(pickle_path, 'rb') as stream:
        foo = pickle.load(stream)
    return foo
