"""Hand crafted training"""
import logging
import os
import random
from datetime import datetime

import numpy as np

from utils import read_json, read_yaml, write_yaml, read_data, load_questions
from memory import Memory

# for reproducibility
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def find_duplicate_head(memory: Memory, observation: list) -> list:
    """Find if there are duplicate heads and return its index.

    Args
    ----
    memory: Memory object
    observation: An observation as a quadruple (i.e., (head, relation, tail, timestamp))

    Returns
    -------
    mem: the memory whose head is the same as that of the observation
        (i.e., (head, relation, tail, timestamp))

    """
    logging.debug("finding if duplicate heads exist ...")
    for mem in memory.entries:
        if mem[0] == observation[0]:
            logging.info(
                f"{mem} has the same head as the observation {observation} !!!"
            )
            return mem

    return None


def find_similar(memory: Memory):
    """Find N episodic memories that can be compressed into one semantic.

    Args
    ----
    memory: Memory object

    Returns
    -------
    episodic_memories: similar episodic memories
    semantic_memory: encoded (compressed) semantic memory.

    """
    logging.debug("looking for episodic entries that can be compressed ...")
    assert memory.type == "episodic"
    # -1 removes the timestamps from the quadruples
    semantic_possibles = [
        [e.split()[-1] for e in entry[:-1]] for entry in memory.entries
    ]
    semantic_possibles = ["_".join(elem) for elem in semantic_possibles]

    def duplicates(mylist, item):
        return [i for i, x in enumerate(mylist) if x == item]

    semantic_possibles = dict(
        (x, duplicates(semantic_possibles, x)) for x in set(semantic_possibles)
    )

    if len(semantic_possibles) == len(memory.entries):
        logging.info("no episodic memories found to be compressible.")
        return None, None
    elif len(semantic_possibles) < len(memory.entries):
        logging.debug("some episodic memories found to be compressible.")

        max_key = max(semantic_possibles, key=lambda k: len(semantic_possibles[k]))
        indexes = semantic_possibles[max_key]

        episodic_memories = map(memory.entries.__getitem__, indexes)
        episodic_memories = list(episodic_memories)
        # sort from the oldest to the latest
        episodic_memories = sorted(episodic_memories, key=lambda x: x[-1])
        semantic_memory = max_key.split("_")
        # The timestamp of this semantic memory is the last observation time of the
        # latest episodic memory.
        semantic_memory.append(episodic_memories[-1][-1])
        assert (len(semantic_memory)) == 4
        for mem in episodic_memories:
            assert len(mem) == 4

        logging.info(
            f"{len(indexes)} episodic memories can be compressed "
            f"into one semantic memory: {semantic_memory}."
        )

        return episodic_memories, semantic_memory
    else:
        raise ValueError


def train_only_episodic(
    policy: dict,
    data: dict,
    capacity: dict,
    save_at: str,
    questions: dict,
    creation_time: int,
) -> None:
    """Train only with a episodic memory system.

    Args
    ----
    policy: e.g., {'FIFO': True, 'NRO': True}
    data: train, val, test data splits in dict
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    save_at: where to save training results.
    questions: questions in train, val, and test splits
    creation_time: the unix time, in seconds, where the agent is first created.

    """
    logging.info(f"episodic only training has started with policy {policy}")

    results = {"train": {}, "val": {}, "test": {}}
    if policy["FIFO"]:

        for split in ["train", "val", "test"]:
            logging.info(f"Running on {split} split ...")

            M_e = Memory("episodic", capacity["episodic"])
            rewards = 0

            for idx, ob in enumerate(data[split]):
                if policy["NRO"]:
                    mem = find_duplicate_head(M_e, ob)
                    if mem is not None:
                        M_e.forget(mem)
                if M_e.is_full:
                    M_e.forget(None)  # FIFO
                else:
                    M_e.add(ob)

                if idx < len(data[split]) - 1:
                    idx_q = idx + 1
                else:
                    idx_q = idx

                question = random.sample(questions[split][idx_q], 1)[0]
                logging.info(f"question: {question}")
                if question in [entry[:-1] for entry in M_e.entries]:
                    rewards += 1
                    logging.debug(f"{question} was in the memory!")

            results[split]["rewards"] = rewards
            results[split]["num_samples"] = len(data[split])

            logging.info(f"results so far: {results}")

    else:
        raise NotImplementedError

    logging.info("episodic only training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "episodic"
    results["episodic_memories"] = M_e.entries
    write_yaml(results, os.path.join(save_at, "results.yaml"))
    logging.info(
        f"training results written at {os.path.join(save_at, 'results.yaml')}!"
    )


def train_only_semantic(
    policy: dict,
    data: dict,
    capacity: dict,
    pretrained_semantic: str,
    save_at: str,
    questions: dict,
    creation_time: int,
) -> None:
    """Train with only semantic memory.

    Args
    ----
    policy: e.g., {'FIFO': True, 'NRO': True}
    data: train, val, test data splits in dict
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    pretrained_semantic: either the path or None.
    save_at: where to save training results.
    questions: questions in train, val, and test splits
    creation_time: the unix time, in seconds, where the agent is first created.

    """
    logging.info(f"semantic only training has started with policy {policy}")

    results = {"train": {}, "val": {}, "test": {}}
    if policy["FIFO"]:

        for split in ["train", "val", "test"]:
            logging.info(f"Running on {split} split ...")

            M_s = Memory("semantic", capacity["semantic"])
            if pretrained_semantic is not None:
                semantics = read_json(pretrained_semantic)
                for head, relation_head in semantics.items():

                    if M_s.is_full:
                        break

                    relation = list(relation_head.keys())[0]
                    tail = relation_head[relation][0]
                    # timestamp is the time when the semantic memory was first stored
                    mem = [head, relation, tail, creation_time]
                    M_s.add(mem)

                M_s.freeze()

            rewards = 0

            for idx, ob in enumerate(data[split]):
                if not M_s.is_frozen:
                    # timestamp is the time when the semantic memory was first stored
                    ob = [
                        ob_.split()[-1] if isinstance(ob_, str) else ob_ for ob_ in ob
                    ]  # split to remove the name

                    if policy["NRO"]:
                        mem = find_duplicate_head(M_s, ob)
                        if mem is not None:
                            M_s.forget(mem)
                    if M_s.is_full:
                        M_s.forget(None)  # FIFO
                    else:
                        M_s.add(ob)

                if idx < len(data[split]) - 1:
                    idx_q = idx + 1
                else:
                    idx_q = idx
                question = random.sample(questions[split][idx_q], 1)[0]
                question = [q.split()[-1] for q in question]
                logging.info(f"question: {question}")
                if question in [entry[:-1] for entry in M_s.entries]:
                    rewards += 1
                    logging.debug(f"{question} was semantic in the memory!")

            results[split]["rewards"] = rewards
            results[split]["num_samples"] = len(data[split])

            logging.info(f"results so far: {results}")

    else:
        raise NotImplementedError

    logging.info("semantic only training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "semantic"
    results["semantic_memories"] = M_s.entries
    write_yaml(results, os.path.join(save_at, "results.yaml"))
    logging.info(
        f"training results written at {os.path.join(save_at, 'results.yaml')}!"
    )


def train_both_episodic_and_semantic(
    policy: dict,
    data: dict,
    capacity: dict,
    pretrained_semantic: str,
    save_at: str,
    questions: dict,
    creation_time: int,
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
    questions: questions in train, val, and test splits
    creation_time: the unix time, in seconds, where the agent is first created.

    """
    logging.info(
        f"both episodic and semantic training has started with policy {policy}"
    )

    results = {"train": {}, "val": {}, "test": {}}
    if policy["FIFO"]:

        for split in ["train", "val", "test"]:
            logging.info(f"Running on {split} split ...")
            M_e = Memory("episodic", capacity["episodic"])
            M_s = Memory("semantic", capacity["semantic"])

            if pretrained_semantic is not None:
                semantics = read_json(pretrained_semantic)
                for head, relation_head in semantics.items():

                    if M_s.is_full:
                        break

                    relation = list(relation_head.keys())[0]
                    tail = relation_head[relation][0]
                    # timestamp is the time when the semantic memory was first stored
                    mem = [head, relation, tail, creation_time]
                    M_s.add(mem)

                M_s.freeze()
                logging.info(
                    f"The semantic memory is pretrained and frozen. The remaining space "
                    f"{capacity['episodic'] + capacity['semantic'] - M_s.capacity} "
                    f"will be used for episodic memory!"
                )
                M_s.capacity = len(M_s.entries)
                M_e.capacity = len(M_e.entries) + (
                    capacity["episodic"] + capacity["semantic"] - M_s.capacity
                )

            rewards = 0

            for idx, ob in enumerate(data[split]):

                if policy["NRO"]:
                    mem = find_duplicate_head(M_e, ob)
                    if mem is not None:
                        M_e.forget(mem)

                if M_e.is_full:
                    episodic_memories, semantic_memory = find_similar(M_e)
                    if episodic_memories is None:
                        logging.info("nothing to be compressed!")
                        M_e.forget(None)  # FIFO
                    else:
                        logging.info(
                            f"{len(episodic_memories)} can be compressed into "
                            f"{semantic_memory}!"
                        )
                        for mem in episodic_memories:
                            M_e.forget(mem)
                        if M_s.is_full:
                            if not M_s.is_frozen:
                                M_s.forget(None)  # FIFO
                        else:
                            if not M_s.is_frozen:
                                M_s.add(semantic_memory)
                else:
                    M_e.add(ob)

                if idx < len(data[split]) - 1:
                    idx_q = idx + 1
                else:
                    idx_q = idx

                question = random.sample(questions[split][idx_q], 1)[0]
                logging.info(f"question: {question}")

                if question in [entry[:-1] for entry in M_e.entries]:
                    rewards += 1
                    logging.debug(f"{question} was in the episodic memory!")
                else:
                    question = [q.split()[-1] for q in question]
                    logging.info(f"question: {question}")

                    if question in [entry[:-1] for entry in M_s.entries]:
                        rewards += 1
                        logging.debug(f"{question} was in the semantic memory!")

            results[split]["rewards"] = rewards
            results[split]["num_samples"] = len(data[split])

            logging.info(f"results so far: {results}")
    else:
        raise NotImplementedError

    logging.info("both episodic and semantic training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "both"
    if pretrained_semantic is not None:
        results["memory_type"] += "_presem"
    results["episodic_memories"] = M_e.entries
    results["semantic_memories"] = M_s.entries
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
    question_path: str,
    creation_time: int,
) -> None:
    """Run training with the given arguments.

    Args
    ----
    memory_type: 'episodic', 'semantic', or 'both'
    policy: e.g., {'FIFO': True, 'NRO': True}
    save_at: where to save training results.
    data_path: path to data.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    pretrained_semantic: either the path or None.
    question_path: path to the questions file.
    creation_time: the unix time, in seconds, where the agent is first created.

    """
    data = read_data(data_path)
    questions = load_questions(question_path)
    if memory_type == "episodic":
        train_only_episodic(policy, data, capacity, save_at, questions, creation_time)
    elif memory_type == "semantic":
        train_only_semantic(
            policy,
            data,
            capacity,
            pretrained_semantic,
            save_at,
            questions,
            creation_time,
        )
    elif memory_type == "both":
        train_both_episodic_and_semantic(
            policy,
            data,
            capacity,
            pretrained_semantic,
            save_at,
            questions,
            creation_time,
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
