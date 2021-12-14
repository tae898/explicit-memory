import random
from copy import deepcopy

from tqdm import tqdm

from memory import EpisodicMemory, SemanticMemory
from memory.environment.generator import OQAGenerator
from memory.utils import write_json, read_yaml
import argparse
import logging
import os

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def sanity_check(results):
    """Check if things add up."""
    for split in ["train", "val", "test"]:
        rewards = 0
        for result in results[split]:
            if result["prediction_hand_crafted"] == result["correct_answer"]:
                rewards += 1

        assert rewards == results["rewards"][split]
        assert len(results[split]) == results["max_history"]


def main(
    seed: int,
    semantic_knowledge_path: str,
    names_path: str,
    max_history: list,
    capacity: list,
    weighting_mode: str,
    commonsense_prob: float,
    save_prefix: str,
    limits: dict
):
    for max_history in tqdm(max_history):
        for capacity_ in tqdm(capacity):

            results = {
                "train": [],
                "val": [],
                "test": [],
                "max_history": max_history,
                "capacity": capacity_,
                "rewards": {"train": None, "val": None, "test": None},
                "accuracy": {"train": None, "val": None, "test": None},
            }

            oqag = OQAGenerator(
                max_history=max_history,
                weighting_mode=weighting_mode,
                commonsense_prob=commonsense_prob,
                semantic_knowledge_path=semantic_knowledge_path,
                names_path=names_path,
                limits=limits,
            )
            for split_idx, split in enumerate(["train", "val", "test"]):
                random.seed(seed + split_idx)
                oqag.reset()
                M_e = EpisodicMemory(capacity_)
                M_s = SemanticMemory(capacity_)

                rewards = 0
                for idx in range(max_history):

                    ob, question_answer = oqag.generate()
                    mem_epi = M_e.ob2epi(ob)
                    M_e.add(mem_epi)
                    if M_e.is_kinda_full:
                        episodic_memories, semantic_memory = M_e.get_similar()
                        if episodic_memories is not None:
                            M_s.add(semantic_memory)
                            if M_s.is_kinda_full:
                                M_s.forget_weakest()
                            for episodic_memory in episodic_memories:
                                M_e.forget(episodic_memory)
                        else:
                            M_e.forget_oldest()

                    qa_epi = question_answer
                    qa_sem = M_s.eq2sq(qa_epi)

                    if M_e.is_answerable(qa_epi):
                        reward, pred, correct_answer = M_e.answer_latest(qa_epi)
                    else:
                        reward, pred, correct_answer = M_s.answer_strongest(qa_sem)

                    rewards += reward
                    results[split].append(
                        {
                            "episodic_memory_system": deepcopy(M_e.entries),
                            "semantic_memory_system": deepcopy(M_s.entries),
                            "question": question_answer[:-1],
                            "prediction_hand_crafted": pred,
                            "correct_answer": correct_answer,
                        }
                    )

                results["rewards"][split] = rewards
                results["accuracy"][split] = rewards / max_history

            sanity_check(results)
            write_json(results, f"{save_prefix}-{max_history}_{capacity_}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create data for the students.")
    parser.add_argument(
        "--config",
        type=str,
        default="data-for-students.yaml",
        help="path to the yaml config.",
    )

    args = read_yaml(parser.parse_args().config)

    logging.info(f"args: {args}")

    main(**args)
