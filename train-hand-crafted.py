"""Hand crafted training"""
import logging
import os
import random
import argparse
from pprint import pformat
from memory.environments import OQAGenerator
from memory.utils import write_json, read_json
from memory import EpisodicMemory, SemanticMemory
from copy import deepcopy

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_only_episodic(
    policy: dict,
    save_at: str,
    names_path: str,
    capacity: dict,
    semantic_knowledge_path: str,
    seed: int,
    max_history: int,
    commonsense_prob: float,
    weighting_mode: str,
    pretrain_semantic: bool,
) -> None:
    """Train only with a episodic memory system.

    Args
    ----
    policy: e.g., {'forget': 'oldest', 'answer': latest}
    save_at: where to save training results.
    names_path: The path to the top 20 human name list.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    semantic_knowledge_path: either the path or None.
    seed: random seed.
    max_history: maximum history of observations.
    weighting_mode: "highest" chooses the one with the highest weight, "weighted"
        chooses all of them by weight, and null chooses every single one of them
        without weighting.
    commonsense_prob: the probability of an observation being covered by a
        commonsense
    pretrain_semantic: whether to pretrain semantic or not.

    """
    logging.info(f"episodic only training has started with policy {policy}")

    results = {"train": {}, "val": {}, "test": {}}
    oqag = OQAGenerator(
        max_history=max_history,
        semantic_knowledge_path=semantic_knowledge_path,
        names_path=names_path,
        weighting_mode=weighting_mode,
        commonsense_prob=commonsense_prob,
    )
    for split_idx, split in enumerate(["train", "val", "test"]):
        # for reproducibility
        random.seed(seed + split_idx)

        logging.info(f"Running on {split} split ...")
        oqag.reset()

        M_e = EpisodicMemory(capacity["episodic"])
        rewards = 0

        for _ in range(max_history):
            ob, question_answer = oqag.generate()
            mem_epi = M_e.ob2epi(ob)
            M_e.add(mem_epi)
            if M_e.is_kinda_full:
                if policy["forget"].lower() == "oldest":
                    M_e.forget_oldest()
                elif policy["forget"].lower() == "random":
                    M_e.forget_random()
                else:
                    raise NotImplementedError

            if policy["answer"].lower() == "latest":
                reward, _, _ = M_e.answer_latest(question_answer)
            elif policy["answer"].lower() == "random":
                reward, _, _ = M_e.answer_random(question_answer)
            else:
                raise NotImplementedError

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["episodic_memories"] = deepcopy(M_e.entries)

        logging.info(f"results so far: {results}")

    logging.info("episodic only training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "episodic"

    config = read_json(os.path.join(save_at, "train-hand-crafted.json"))
    results.update(config)

    write_json(results, os.path.join(save_at, "results.json"))
    logging.info(
        f"training results written at {os.path.join(save_at, 'results.json')}!"
    )


def train_only_semantic(
    policy: dict,
    save_at: str,
    names_path: str,
    capacity: dict,
    semantic_knowledge_path: str,
    seed: int,
    max_history: int,
    commonsense_prob: float,
    weighting_mode: str,
    pretrain_semantic: bool,
) -> None:
    """Train with only semantic memory.

    Args
    ----
    policy: e.g., {'forget': 'weakest', 'answer': strongest}
    save_at: where to save training results.
    names_path: The path to the top 20 human name list.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    semantic_knowledge_path: either the path or None.
    max_history: maximum history of observations.
    seed: random seed.
    max_history: maximum history of observations.
    weighting_mode: "highest" chooses the one with the highest weight, "weighted"
        chooses all of them by weight, and null chooses every single one of them
        without weighting.
    commonsense_prob: the probability of an observation being covered by a
        commonsense
    pretrain_semantic: whether to pretrain semantic or not.

    """
    logging.info(f"semantic only training has started with policy {policy}")

    results = {"train": {}, "val": {}, "test": {}}
    oqag = OQAGenerator(
        max_history=max_history,
        semantic_knowledge_path=semantic_knowledge_path,
        names_path=names_path,
        weighting_mode=weighting_mode,
        commonsense_prob=commonsense_prob,
    )

    for split_idx, split in enumerate(["train", "val", "test"]):
        # for reproducibility
        random.seed(seed + split_idx)

        logging.info(f"Running on {split} split ...")
        oqag.reset()

        M_s = SemanticMemory(capacity["semantic"])
        if pretrain_semantic:
            free_space = M_s.pretrain_semantic(
                semantic_knowledge_path, weighting_mode=weighting_mode
            )

        rewards = 0
        for _ in range(max_history):
            ob, question_answer = oqag.generate()
            mem_sem = M_s.ob2sem(ob)
            if not M_s.is_frozen:
                M_s.add(mem_sem)
                if M_s.is_kinda_full and (not M_s.is_frozen):
                    if policy["forget"].lower() == "weakest":
                        M_s.forget_weakest()
                    elif policy["forget"].lower() == "random":
                        M_s.forget_random()
                    else:
                        raise NotImplementedError

            question = M_s.eq2sq(question_answer)

            if policy["answer"].lower() == "strongest":
                reward, _, _ = M_s.answer_strongest(question)
            elif policy["answer"].lower() == "random":
                reward, _, _ = M_s.answer_random(question)
            else:
                raise NotImplementedError

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["semantic_memories"] = deepcopy(M_s.entries)

        logging.info(f"results so far: {results}")

    logging.info("semantic only training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "semantic"
    results["pretrain_semantic"] = pretrain_semantic
    config = read_json(os.path.join(save_at, "train-hand-crafted.json"))
    results.update(config)
    write_json(results, os.path.join(save_at, "results.json"))
    logging.info(
        f"training results written at {os.path.join(save_at, 'results.json')}!"
    )


def train_both_episodic_and_semantic(
    policy: dict,
    save_at: str,
    names_path: str,
    capacity: dict,
    semantic_knowledge_path: str,
    seed: int,
    max_history: int,
    commonsense_prob: float,
    weighting_mode: str,
    pretrain_semantic: str,
) -> None:
    """Train with both episodic and semantic memory.

    Args
    ----

    policy: e.g., {'episodic': {'forget': 'oldest', 'answer': latest},
        'semantic': {'forget': 'weakest', 'answer': strongest}}
    save_at: where to save training results.
    names_path: The path to the top 20 human name list.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    semantic_knowledge_path: either the path or None.
    seed: random seed.
    max_history: maximum history of observations.
    weighting_mode: "highest" chooses the one with the highest weight, "weighted"
        chooses all of them by weight, and null chooses every single one of them
        without weighting.
    commonsense_prob: the probability of an observation being covered by a
        commonsense
    pretrain_semantic: whether to pretrain semantic or not.

    """
    logging.info(
        f"both episodic and semantic training has started with policy {policy}"
    )

    results = {"train": {}, "val": {}, "test": {}}
    oqag = OQAGenerator(
        max_history=max_history,
        semantic_knowledge_path=semantic_knowledge_path,
        names_path=names_path,
        weighting_mode=weighting_mode,
        commonsense_prob=commonsense_prob,
    )
    for split_idx, split in enumerate(["train", "val", "test"]):
        # for reproducibility
        random.seed(seed + split_idx)

        logging.info(f"Running on {split} split ...")
        oqag.reset()

        M_e = EpisodicMemory(capacity["episodic"])
        M_s = SemanticMemory(capacity["semantic"])

        if pretrain_semantic:
            free_space = M_s.pretrain_semantic(
                semantic_knowledge_path, weighting_mode=weighting_mode
            )
            M_e.increase_capacity(free_space)
            assert (M_e.capacity + M_s.capacity) == (
                capacity["episodic"] + capacity["semantic"]
            )

        rewards = 0
        for _ in range(max_history):
            ob, question_answer = oqag.generate()
            mem_epi = M_e.ob2epi(ob)
            M_e.add(mem_epi)
            if M_s.is_frozen:
                if M_e.is_kinda_full:
                    if policy["episodic"]["forget"].lower() == "oldest":
                        M_e.forget_oldest()
                    elif policy["episodic"]["forget"].lower() == "random":
                        M_e.forget_random()
                    else:
                        raise NotImplementedError
            else:
                if M_e.is_kinda_full:
                    episodic_memories, semantic_memory = M_e.get_similar()
                    if episodic_memories is not None:
                        M_s.add(semantic_memory)
                        if M_s.is_kinda_full:
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

            question_epi = question_answer
            question_sem = M_s.eq2sq(question_epi)

            if policy["episodic"]["answer"].lower() == "latest":
                if M_e.is_answerable(question_epi):
                    reward, _, _ = M_e.answer_latest(question_epi)
                else:
                    if policy["semantic"]["answer"].lower() == "strongest":
                        reward, _, _ = M_s.answer_strongest(question_sem)
                    elif policy["semantic"]["answer"].lower() == "random":
                        reward, _, _ = M_s.answer_random(question_sem)
                    else:
                        raise NotImplementedError
            elif policy["episodic"]["answer"].lower() == "random":
                reward, _, _ = M_e.answer_random(question_epi)
            else:
                raise NotImplementedError

            rewards += reward

        results[split]["rewards"] = rewards
        results[split]["episodic_memories"] = deepcopy(M_e.entries)
        results[split]["semantic_memories"] = deepcopy(M_s.entries)

        logging.info(f"results so far: {results}")

    logging.info("both episodic and semantic training done!")

    logging.debug("writing training results ...")
    results["policy"] = policy
    results["capacity"] = capacity
    results["memory_type"] = "both"
    results["pretrain_semantic"] = pretrain_semantic
    config = read_json(os.path.join(save_at, "train-hand-crafted.json"))
    results.update(config)
    write_json(results, os.path.join(save_at, "results.json"))
    logging.info(
        f"training results written at {os.path.join(save_at, 'results.json')}!"
    )


def main(
    memory_type: str,
    policy: dict,
    save_at: str,
    names_path: str,
    capacity: dict,
    semantic_knowledge_path: str,
    seed: int,
    max_history: int,
    commonsense_prob: float,
    weighting_mode: str,
    pretrain_semantic: str,
) -> None:
    """Run training with the given arguments.

    Args
    ----
    memory_type: 'episodic', 'semantic', or 'both'
    policy: e.g., {'episodic': {'forget': 'oldest', 'answer': latest},
        'semantic': {'forget': 'weakest', 'answer': strongest}}
    save_at: where to save training results.
    names_path: The path to the top 20 human name list.
    capacity: memory capacity for episodic and semantic
        e.g., {'episodic': 46, 'semantic': 0}
    semantic_knowledge_path: either the path or None.
    seed: random seed.
    max_history: maximum history of observations.
    weighting_mode: "highest" chooses the one with the highest weight, "weighted"
        chooses all of them by weight, and null chooses every single one of them
        without weighting.
    commonsense_prob: the probability of an observation being covered by a
        commonsense
    pretrain_semantic: whether to pretrain semantic or not.

    """

    if memory_type == "episodic":
        train_only_episodic(
            policy["episodic"],
            save_at,
            names_path,
            capacity,
            semantic_knowledge_path,
            seed,
            max_history,
            commonsense_prob,
            weighting_mode,
            pretrain_semantic,
        )
    elif memory_type == "semantic":
        train_only_semantic(
            policy["semantic"],
            save_at,
            names_path,
            capacity,
            semantic_knowledge_path,
            seed,
            max_history,
            commonsense_prob,
            weighting_mode,
            pretrain_semantic,
        )
    elif memory_type == "both":
        train_both_episodic_and_semantic(
            policy,
            save_at,
            names_path,
            capacity,
            semantic_knowledge_path,
            seed,
            max_history,
            commonsense_prob,
            weighting_mode,
            pretrain_semantic,
        )
    else:
        raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train handcrafted policies")
    parser.add_argument("--config", type=str, default="train-hand-crafted.json")
    args = parser.parse_args()

    config = read_json(args.config)
    logging.info(f"\nArguments\n---------\n{pformat(config,indent=4, width=1)}\n")

    logging.debug(f"Training results will be saved at {config['save_at']}")
    main(**config)
