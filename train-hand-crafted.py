"""Hand crafted training"""
import logging
import os
import random
from datetime import datetime

import numpy as np

from utils import read_json, read_yaml, write_yaml

# for reproducibility
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Memory:
    """Memory (episodic or semantic) class"""

    def __init__(self, memory_type: str, capacity: dict) -> None:
        """
        Args
        ----
        memory_type: either episodic or semantic.
        capacity: memory capacity for episodic and semantic
            e.g., {'episodic': 46, 'semantic': 0}
        """
        logging.debug(
            f"instantiating a {memory_type} memory object with size {capacity} ..."
        )

        assert memory_type in ["episodic", "semantic"]
        self.memory_type = memory_type
        self.memory = []
        self.capacity = capacity
        self._frozen = False

        logging.debug(f"{memory_type} memory object with size {capacity} instantiated!")

    def __repr__(self):
        from pprint import pformat

        return pformat(vars(self), indent=4, width=1)

    def remove(self, idx: int):
        """Remove idx'th memory.
        Args
        ----
        idx: index
        """
        assert not self._frozen
        logging.debug(f"Removing idx'th memory ...")
        self.memory.pop(idx)
        logging.info(f"idx'th memory removed!")

    def add(self, elem: list):
        """Append elem to the memory."""
        assert not self._frozen
        if len(self.memory) >= self.capacity[self.memory_type]:
            logging.error(f"memory is full! can't add more.")
            raise ValueError(f"memory is full! can't add more.")

        logging.debug(f"Adding a new memory entry {elem} ...")
        self.memory.append(elem)
        logging.info(f"memory entry {elem} added!")

    @property
    def is_full(self):
        """Return true if full."""
        return len(self.memory) == self.capacity[self.memory_type]

    def find_duplicate_head(self, elem: list) -> int:
        """Find if there are duplicate heads and return its index."""
        logging.debug(f"finding if duplicate heads exist ...")
        for idx, mem in enumerate(self.memory):
            if mem[0] == elem[0]:
                logging.info(f"{mem} has the same head as the new obs {elem} !!!")
                return idx

        return None

    @property
    def is_frozen(self):
        """Is frozen?"""
        return self._frozen

    def freeze(self):
        """Freeze the memory so that nothing can be added / deleted."""
        self._frozen = True

    def find_similar(self):
        """Find N episodic entries that can be compressed into one semantic."""
        logging.debug(f"looking for episodic entries that can be compressed ...")
        assert self.memory_type == "episodic"
        semantic_possibles = [
            [entry.split()[-1] for entry in entry] for entry in self.memory
        ]
        semantic_possibles = ["_".join(elem) for elem in semantic_possibles]

        def duplicates(mylist, item):
            return [i for i, x in enumerate(mylist) if x == item]

        semantic_possibles = dict(
            (x, duplicates(semantic_possibles, x)) for x in set(semantic_possibles)
        )

        if len(semantic_possibles) == len(self.memory):
            logging.info(f"no episodic memories found to be compressible.")
            return None, None
        elif len(semantic_possibles) < len(self.memory):
            logging.debug(f"some episodic memories found to be compressible.")

            max_key = max(semantic_possibles, key=lambda k: len(semantic_possibles[k]))
            indexes = semantic_possibles[max_key]
            semantic_memory = max_key.split("_")

            logging.warning(
                f"{len(indexes)} episodic memories can be compressed "
                f"into one semantic memory: {semantic_memory}."
            )

            return indexes, semantic_memory
        else:
            raise ValueError


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


def train_only_episodic(policy: dict, data: dict, capacity: dict, save_at: str) -> None:
    """Train with only episodic memory.

    Args
    ----
    policy: e.g., {'forget': FIFO, 'duplicate_heads': False}
    data: train, val, test data splits in dict
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    save_at: where to save training results.
    """
    logging.info(f"episodic only training has started with policy {policy}")

    results = {"train": {}, "val": {}, "test": {}}
    if policy["forget"].upper() == "FIFO":

        for split in ["train", "val", "test"]:
            logging.info(f"Running on {split} split ...")

            M_e = Memory("episodic", capacity)
            rewards = 0
            sample_from = []

            for idx, obs in enumerate(data[split]):
                if not policy["duplicate_heads"]:
                    j = M_e.find_duplicate_head(obs)
                    if j is not None:
                        M_e.remove(j)
                if M_e.is_full:
                    M_e.remove(0)  # FIFO
                else:
                    M_e.add(obs)

                for _ in range(idx + 1):
                    sample_from.append(obs)

                question = random.sample(sample_from, 1)[0]
                if question in M_e.memory:
                    rewards += 1
                    logging.debug(f"{question} was in the memory!")

            results[split]["rewards"] = rewards
            results[split]["num_samples"] = len(data[split])

            logging.info(f"results so far: {results}")

    else:
        raise NotImplementedError

    logging.info(f"episodic only training done!")

    logging.debug(f"writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "episodic"
    write_yaml(results, os.path.join(save_at, "results.yaml"))
    logging.info(
        f"training results written at {os.path.join(save_at, 'results.yaml')}!"
    )


def train_only_semantic(
    policy: dict, data: dict, capacity: dict, pretrained_semantic: str, save_at: str
) -> None:
    """Train with only semantic memory.

    Args
    ----
    policy: e.g., {'forget': FIFO, 'duplicate_heads': False}
    data: train, val, test data splits in dict
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    pretrained_semantic: either the path or None.
    save_at: where to save training results.
    """
    logging.info(f"semantic only training has started with policy {policy}")

    results = {"train": {}, "val": {}, "test": {}}
    if policy["forget"].upper() == "FIFO":

        for split in ["train", "val", "test"]:
            logging.info(f"Running on {split} split ...")

            M_s = Memory("semantic", capacity)
            if pretrained_semantic is not None:
                for head, relation_head in read_json(pretrained_semantic).items():

                    relation = list(relation_head.keys())[0]
                    tail = relation_head[relation][0]
                    obs = [head, relation, tail]

                    if M_s.is_full:
                        break

                    M_s.add(obs)
                M_s.freeze()

            rewards = 0
            sample_from = []

            for idx, obs in enumerate(data[split]):

                if not M_s.is_frozen:
                    obs = [ob.split()[-1] for ob in obs]
                    if not policy["duplicate_heads"]:
                        j = M_s.find_duplicate_head(obs)
                        if j is not None:
                            M_s.remove(j)
                    if M_s.is_full:
                        M_s.remove(0)  # FIFO
                    else:
                        M_s.add(obs)

                for _ in range(idx + 1):
                    sample_from.append(obs)

                question = random.sample(sample_from, 1)[0]
                if question in M_s.memory:
                    rewards += 1
                    logging.debug(f"{question} was in the memory!")

            results[split]["rewards"] = rewards
            results[split]["num_samples"] = len(data[split])

            logging.info(f"results so far: {results}")

    else:
        raise NotImplementedError

    logging.info(f"semantic only training done!")

    logging.debug(f"writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "semantic"
    write_yaml(results, os.path.join(save_at, "results.yaml"))
    logging.info(
        f"training results written at {os.path.join(save_at, 'results.yaml')}!"
    )


def train_both_episodic_and_semantic(
    policy: dict, data: dict, capacity: dict, pretrained_semantic: str, save_at: str
) -> None:
    """Train with both episodic and semantic memory.

    Args
    ----
    policy: e.g., FIFO
    data: train, val, test data splits in dict.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    pretrained_semantic: either the path or None.
    save_at: where to save training results.
    """
    logging.info(
        f"both episodic and semantic training has started with policy {policy}"
    )

    results = {"train": {}, "val": {}, "test": {}}
    if policy["forget"].upper() == "FIFO":

        for split in ["train", "val", "test"]:
            logging.info(f"Running on {split} split ...")
            M_e = Memory("episodic", capacity)
            M_s = Memory("semantic", capacity)

            if pretrained_semantic is not None:
                for head, relation_head in read_json(pretrained_semantic).items():

                    relation = list(relation_head.keys())[0]
                    tail = relation_head[relation][0]
                    obs = [head, relation, tail]

                    if M_s.is_full:
                        break

                    M_s.add(obs)
                M_s.freeze()

            rewards = 0
            sample_from = []

            for idx, obs in enumerate(data[split]):

                if not policy["duplicate_heads"]:
                    j = M_e.find_duplicate_head(obs)
                    if j is not None:
                        M_e.remove(j)
                if M_e.is_full:
                    js, semantic_memory = M_e.find_similar()
                    if js is None:
                        logging.info(f"nothing to be compressed!")
                        M_e.remove(0)  # FIFO
                    else:
                        logging.info(
                            f"{len(js)} can be compressed into {semantic_memory}!"
                        )
                        for j in sorted(js, reverse=True):
                            M_e.remove(j)
                        if M_s.is_full:
                            if not M_s.is_frozen:
                                M_s.remove(0)  # FIFO
                        else:
                            if not M_s.is_frozen:
                                M_s.add(semantic_memory)
                else:
                    M_e.add(obs)

                for _ in range(idx + 1):
                    sample_from.append(obs)

                question = random.sample(sample_from, 1)[0]
                if question in M_e.memory:
                    rewards += 1
                    logging.debug(f"{question} was in the episodic memory!")
                else:
                    question = [question.split()[-1] for question in question]
                    if question in M_s.memory:
                        rewards += 1
                        logging.debug(f"{question} was in the semantic memory!")

            results[split]["rewards"] = rewards
            results[split]["num_samples"] = len(data[split])

            logging.info(f"results so far: {results}")
    else:
        raise NotImplementedError

    logging.info(f"both episodic and semantic training done!")

    logging.debug(f"writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "both"
    if pretrained_semantic is not None:
        results["memory_type"] += "_presem"
    write_yaml(results, os.path.join(save_at, "results.yaml"))
    logging.info(
        f"training results written at {os.path.join(save_at, 'results.yaml')}!"
    )


def main(
    memory_type: str,
    policy: dict,
    save_at: str,
    data_path: str,
    capacity: dict,
    pretrained_semantic: str,
) -> None:
    """Run training with the given arguments.

    Args
    ----
    memory_type: 'episodic', 'semantic', or 'both'
    policy: e.g., {'forget': FIFO, 'duplicate_heads': False}
    save_at: where to save training results.
    data_path: path to data.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    pretrained_semantic: either the path or None.

    """
    data = read_data(data_path)
    if memory_type == "episodic":
        train_only_episodic(policy, data, capacity, save_at)
    elif memory_type == "semantic":
        train_only_semantic(policy, data, capacity, pretrained_semantic, save_at)
    elif memory_type == "both":
        train_both_episodic_and_semantic(
            policy, data, capacity, pretrained_semantic, save_at
        )
    else:
        raise ValueError


if __name__ == "__main__":
    config = read_yaml("./train-hand-crafted.yaml")
    print("Arguments:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    config["save_at"] = os.path.join(
        config["save_at"], datetime.now().strftime(r"%m%d_%H%M%S")
    )
    os.makedirs(config["save_at"], exist_ok=True)

    logging.debug(f"Training results will be saved at {config['save_at']}")
    write_yaml(config, os.path.join(config["save_at"], "train-hand-crafted.yaml"))
    main(**config)
