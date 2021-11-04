"""Hand crafted training"""
import logging
import os
import random
from datetime import datetime

import numpy as np

from utils import read_json, read_yaml, write_yaml, read_data
from memory import (
    Memory,
    load_questions,
    select_a_question,
    answer_NRO,
    answer_random,
    ob2sem,
)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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

    for split in ["train", "val", "test"]:
        logging.info(f"Running on {split} split ...")

        M_e = Memory("episodic", capacity["episodic"])
        rewards = 0

        for step, ob in enumerate(data[split]):
            if M_e.is_full:
                if policy["FIFO"]:
                    M_e.forget_FIFO()
                else:
                    M_e.forget_random()

            M_e.add(ob)

            question = select_a_question(step, data, questions, split)

            if policy["NRO"]:
                if not M_e.is_empty:
                    reward = answer_NRO(question, M_e)
                else:
                    reward = -1
            else:
                if not M_e.is_empty:
                    reward = answer_random(question, M_e)
                else:
                    reward = -1

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["num_samples"] = len(data[split])

        logging.info(f"results so far: {results}")

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

    for split in ["train", "val", "test"]:
        logging.info(f"Running on {split} split ...")

        M_s = Memory("semantic", capacity["semantic"])
        if pretrained_semantic is not None:
            free_space = M_s.pretrain_semantic(pretrained_semantic, creation_time)

        rewards = 0
        for step, ob in enumerate(data[split]):
            mem = ob2sem(ob)
            if M_s.is_full and (not M_s.is_frozen):
                if policy["FIFO"]:
                    M_s.forget_FIFO()
                else:
                    M_s.forget_random()

            M_s.add(mem)

            question = select_a_question(step, data, questions, split)
            # this is a hack since in our simpliest world, observations, memories,
            # and even questions have almost the same format.
            question = ob2sem(question)

            if policy["NRO"]:
                if not M_s.is_empty:
                    reward = answer_NRO(question, M_s)
                else:
                    reward = -1
            else:
                if not M_s.is_empty:
                    reward = answer_random(question, M_s)
                else:
                    reward = -1

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["num_samples"] = len(data[split])

        logging.info(f"results so far: {results}")

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

    for split in ["train", "val", "test"]:
        logging.info(f"Running on {split} split ...")
        M_e = Memory("episodic", capacity["episodic"])
        M_s = Memory("semantic", capacity["semantic"])

        if pretrained_semantic is not None:
            free_space = M_s.pretrain_semantic(pretrained_semantic, creation_time)
            M_e.capacity += free_space

        rewards = 0
        for step, ob in enumerate(data[split]):
            if M_s.is_frozen:
                if M_e.is_full:
                    if policy["FIFO"]:
                        M_e.forget_FIFO()
                    else:
                        M_e.forget_random()
                M_e.add(ob)
            else:
                if M_e.is_full:
                    episodic_memories, semantic_memory = M_e.get_similar()
                    if episodic_memories is not None:
                        if M_s.is_full:
                            if policy["FIFO"]:
                                M_s.forget_FIFO()
                            else:
                                M_s.forget_random()
                        M_s.add(semantic_memory)
                        for epi_mem in episodic_memories:
                            M_e.forget(epi_mem)
                    else:
                        if policy["FIFO"]:
                            M_e.forget_FIFO()
                        else:
                            M_e.forget_random()
                M_e.add(ob)

            question_epi = select_a_question(step, data, questions, split)
            question_sem = ob2sem(question_epi)

            if policy["NRO"]:
                if not M_e.is_empty:
                    reward_epi = answer_NRO(question_epi, M_e)
                else:
                    reward_epi = -1
                if not M_s.is_empty:
                    reward_sem = answer_NRO(question_sem, M_s)
                else:
                    reward_sem = -1
                reward = max(reward_epi, reward_sem)
            else:
                if not M_e.is_empty:
                    reward_epi = answer_random(question_epi, M_e)
                else:
                    reward_epi = -1
                if not M_s.is_empty:
                    reward_sem = answer_random(question_sem, M_s)
                else:
                    reward_sem = -1
                reward = max(reward_epi, reward_sem)

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["num_samples"] = len(data[split])

        logging.info(f"results so far: {results}")

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
    seed: int,
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
    seed: random seed.

    """
    # for reproducibility
    random.seed(seed)
    np.random.seed(seed)

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
