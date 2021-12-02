import random
import numpy as np
from copy import deepcopy

from tqdm import tqdm

from memory import EpisodicMemory, SemanticMemory
from memory.environments import OQAGenerator
from memory.utils import write_json

seed = 42


def sanity_check(results):
    """Check if things add up."""
    for split in ["val", "test"]:
        rewards = 0
        for result in results[split]:
            if result["prediction_hand_crafted"] == result["correct_answer"]:
                rewards += 1

        assert rewards == results["rewards"][split]
        assert len(results[split]) == results["max_history"]


for max_history in tqdm([128]):
    for capacity in [1, 2, 4, 8, 16, 32, 64]:

        results = {
            "val": [],
            "test": [],
            "max_history": max_history,
            "capacity": capacity,
            "rewards": {"val": None, "test": None},
            "accuracy": {"val": None, "test": None},
        }

        oqag = OQAGenerator(
            max_history=max_history,
            weighting_mode="highest",
            commonsense_prob=0.5,
            limits={"heads": 10, "tails": 1, "names": 5, "allow_spaces": False},
        )
        for split_idx, split in enumerate(["val", "test"]):
            random.seed(seed + split_idx)
            np.random.seed(seed + split_idx)
            oqag.reset()
            M_e = EpisodicMemory(capacity)
            M_s = SemanticMemory(capacity)

            rewards = 0
            for idx in range(max_history):

                ob, question_answer = oqag.generate()
                ob[-1] += 86400 * idx + 1000000000
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
        write_json(results, f"{max_history}_{capacity}.json")
