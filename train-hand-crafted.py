"""Hand crafted training"""
import logging
import os
import random
import argparse
import numpy as np
from pprint import pformat

from utils import (
    read_yaml,
    write_yaml,
    read_data,
    load_questions,
    select_a_question,
)
from memory import EpisodicMemory, SemanticMemory

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
) -> None:
    """Train only with a episodic memory system.

    Args
    ----
    policy: e.g., {'forget': 'oldest', 'answer': newest}
    data: train, val, test data splits in dict
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    save_at: where to save training results.
    questions: questions in train, val, and test splits

    """
    logging.info(f"episodic only training has started with policy {policy}")

    results = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        logging.info(f"Running on {split} split ...")

        M_e = EpisodicMemory(capacity["episodic"])
        rewards = 0

        for step, ob in enumerate(data[split]):
            mem_epi = M_e.ob2epi(ob)
            M_e.add(mem_epi)
            if M_e.is_full:
                if policy["forget"].lower() == "oldest":
                    M_e.forget_oldest()
                elif policy["forget"].lower() == "random":
                    M_e.forget_random()
                else:
                    raise NotImplementedError

            question = select_a_question(step, data, questions, split)

            if policy["answer"].lower() == "newest":
                reward = M_e.answer_newest(question)
            elif policy["answer"].lower() == "random":
                reward = M_e.answer_random(question)
            else:
                raise NotImplementedError

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["num_samples"] = len(data[split])
        results[split]["episodic_memories"] = M_e.entries

        logging.info(f"results so far: {results}")

    logging.info("episodic only training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "episodic"
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
) -> None:
    """Train with only semantic memory.

    Args
    ----
    policy: e.g., {'forget': 'weakest', 'answer': strongest}
    data: train, val, test data splits in dict
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    pretrained_semantic: either the path or None.
    save_at: where to save training results.
    questions: questions in train, val, and test splits

    """
    logging.info(f"semantic only training has started with policy {policy}")

    results = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        logging.info(f"Running on {split} split ...")

        M_s = SemanticMemory(capacity["semantic"])
        if pretrained_semantic is not None:
            free_space = M_s.pretrain_semantic(pretrained_semantic)

        rewards = 0
        for step, ob in enumerate(data[split]):
            mem_sem = M_s.ob2sem(ob)
            if not M_s.is_frozen:
                M_s.add(mem_sem)
                if M_s.is_full and (not M_s.is_frozen):
                    if policy["forget"].lower() == "weakest":
                        M_s.forget_weakest()
                    elif policy["forget"].lower() == "random":
                        M_s.forget_random()
                    else:
                        raise NotImplementedError

            question = select_a_question(step, data, questions, split)
            question = M_s.eq2sq(question)

            if policy["answer"].lower() == "strongest":
                reward = M_s.answer_strongest(question)
            elif policy["answer"].lower() == "random":
                reward = M_s.answer_random(question)
            else:
                raise NotImplementedError

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["num_samples"] = len(data[split])
        results[split]["semantic_memories"] = M_s.entries

        logging.info(f"results so far: {results}")

    logging.info("semantic only training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "semantic"
    if pretrained_semantic is not None:
        results["pretrained_semantic"] = True
    else:
        results["pretrained_semantic"] = False
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
) -> None:
    """Train with both episodic and semantic memory.

    Args
    ----
    policy: e.g., {'episodic': {'forget': 'oldest', 'answer': newest},
        'semantic': {'forget': 'weakest', 'answer': strongest}}
    data: train, val, test data splits in dict.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    pretrained_semantic: either the path or None.
    save_at: where to save training results.
    questions: questions in train, val, and test splits

    """
    logging.info(
        f"both episodic and semantic training has started with policy {policy}"
    )

    results = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        logging.info(f"Running on {split} split ...")
        M_e = EpisodicMemory(capacity["episodic"])
        M_s = SemanticMemory(capacity["semantic"])

        if pretrained_semantic is not None:
            free_space = M_s.pretrain_semantic(pretrained_semantic)
            M_e.increase_capacity(free_space)
            assert (M_e.capacity + M_s.capacity) == (
                capacity["episodic"] + capacity["semantic"]
            )

        rewards = 0
        for step, ob in enumerate(data[split]):
            mem_epi = M_e.ob2epi(ob)
            M_e.add(mem_epi)
            if M_s.is_frozen:
                if M_e.is_full:
                    if policy["episodic"]["forget"].lower() == "oldest":
                        M_e.forget_oldest()
                    elif policy["episodic"]["forget"].lower() == "random":
                        M_e.forget_random()
                    else:
                        raise NotImplementedError
            else:
                if M_e.is_full:
                    episodic_memories, semantic_memory = M_e.get_similar()
                    if episodic_memories is not None:
                        M_s.add(semantic_memory)
                        if M_s.is_full:
                            if policy["semantic"]["forget"].lower() == "weakest":
                                M_s.forget_weakest()
                            elif policy["semantic"]["forget"].lower() == "random":
                                M_s.forget_random()
                            else:
                                raise NotImplementedError

                        for episodic_memory in episodic_memories:
                            M_e.forget(episodic_memory)
                    else:
                        if policy["episodic"]["forget"].lower() == "oldest":
                            M_e.forget_oldest()
                        elif policy["episodic"]["forget"].lower() == "random":
                            M_e.forget_random()
                        else:
                            raise NotImplementedError

            question_epi = select_a_question(step, data, questions, split)
            question_sem = M_s.eq2sq(question_epi)

            if policy["episodic"]["answer"].lower() == "newest":
                reward_epi = M_e.answer_newest(question_epi)
            elif policy["episodic"]["answer"].lower() == "random":
                reward_epi = M_e.answer_random(question_epi)
            else:
                raise NotImplementedError

            if policy["semantic"]["answer"].lower() == "strongest":
                reward_sem = M_s.answer_strongest(question_sem)
            elif policy["semantic"]["answer"].lower() == "random":
                reward_sem = M_s.answer_random(question_sem)
            else:
                raise NotImplementedError

            reward = max(reward_epi, reward_sem)

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["num_samples"] = len(data[split])
        results[split]["episodic_memories"] = M_e.entries
        results[split]["semantic_memories"] = M_s.entries

        logging.info(f"results so far: {results}")

    logging.info("both episodic and semantic training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "both"
    if pretrained_semantic is not None:
        results["pretrained_semantic"] = True
    else:
        results["pretrained_semantic"] = False
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
    seed: int,
) -> None:
    """Run training with the given arguments.

    Args
    ----
    memory_type: 'episodic', 'semantic', or 'both'
    policy: e.g., {'episodic': {'forget': 'oldest', 'answer': newest},
        'semantic': {'forget': 'weakest', 'answer': strongest}}
    save_at: where to save training results.
    data_path: path to data.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    pretrained_semantic: either the path or None.
    question_path: path to the questions file.
    seed: random seed.

    """
    # for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    data = read_data(data_path)
    questions = load_questions(question_path)
    if memory_type == "episodic":
        train_only_episodic(policy["episodic"], data, capacity, save_at, questions)
    elif memory_type == "semantic":
        train_only_semantic(
            policy["semantic"], data, capacity, pretrained_semantic, save_at, questions
        )
    elif memory_type == "both":
        train_both_episodic_and_semantic(
            policy, data, capacity, pretrained_semantic, save_at, questions
        )
    else:
        raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train handcrafted policies")
    parser.add_argument("--config", type=str, default="train-hand-crafted.yaml")
    args = parser.parse_args()

    config = read_yaml(args.config)
    logging.info(f"\nArguments\n---------\n{pformat(config,indent=4, width=1)}\n")

    logging.debug(f"Training results will be saved at {config['save_at']}")
    main(**config)
