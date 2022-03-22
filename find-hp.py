import logging

logger = logging.getLogger()
logger.disabled = True

import argparse
import random
from itertools import count

import numpy as np
from tqdm import tqdm

from memory import EpisodicMemory, SemanticMemory
from memory.environment import RoomEnv
from memory.utils import seed_everything, write_json


def episodic(env_params, caps, seed):

    seed_everything(seed)
    env = RoomEnv(**env_params)

    results = []
    for capacity in caps:
        rewards = {}
        for forget_policy in ["oldest", "random"]:
            for answer_policy in ["latest", "random"]:

                rewards[f"{forget_policy}_{answer_policy}"] = 0
                M_e = EpisodicMemory(capacity)
                ob, question = env.reset()
                ob = ob[0]

                M_e.add(EpisodicMemory.ob2epi(ob))

                for t in count():
                    if M_e.is_full:
                        if forget_policy == "random":
                            M_e.forget_random()
                        else:
                            M_e.forget_oldest()

                    if answer_policy == "latest":
                        pred, _ = M_e.answer_latest(question)
                    else:
                        pred, _ = M_e.answer_random(question)

                    (ob, question), reward, done, info = env.step(pred)
                    ob = ob[0]

                    M_e.add(EpisodicMemory.ob2epi(ob))

                    rewards[f"{forget_policy}_{answer_policy}"] += reward
                    if done:
                        break

        rewards["info"] = {"seed": seed, "capacity": capacity}
        results.append(rewards)

    return results


def semantic(env_params, caps, seed):

    seed_everything(seed)
    env = RoomEnv(**env_params)

    results = []
    for capacity in caps:
        rewards = {}
        for forget_policy in ["weakest", "random"]:
            for answer_policy in ["strongest", "random"]:

                rewards[f"{forget_policy}_{answer_policy}"] = 0
                M_s = SemanticMemory(capacity)
                ob, question = env.reset()
                ob = ob[0]
                M_s.add(SemanticMemory.ob2sem(ob))

                for t in count():
                    if M_s.is_full:
                        if forget_policy == "random":
                            M_s.forget_random()
                        elif forget_policy == "weakest":
                            M_s.forget_weakest()
                        else:
                            raise ValueError

                    if answer_policy == "strongest":
                        pred, _ = M_s.answer_strongest(question)
                    else:
                        pred, _ = M_s.answer_random(question)

                    (ob, question), reward, done, info = env.step(pred)
                    ob = ob[0]

                    M_s.add(SemanticMemory.ob2sem(ob))

                    rewards[f"{forget_policy}_{answer_policy}"] += reward

                    if done:
                        break

        rewards["info"] = {"seed": seed, "capacity": capacity}
        results.append(rewards)

    return results


def episodic_semantic(env_params, caps, seed):

    seed_everything(seed)
    env = RoomEnv(**env_params)

    results = []
    for capacity in caps:
        rewards = {}
        for forget_policy in ["generalize", "random"]:
            for answer_policy in ["episem", "random"]:

                rewards[f"{forget_policy}_{answer_policy}"] = 0
                M_e = EpisodicMemory(capacity // 2)
                M_s = SemanticMemory(capacity // 2)

                ob, question = env.reset()
                ob = ob[0]
                if forget_policy == "random":
                    if random.random() < 0.5:
                        M_e.add(EpisodicMemory.ob2epi(ob))
                    else:
                        M_s.add(SemanticMemory.ob2sem(ob))
                else:
                    M_e.add(EpisodicMemory.ob2epi(ob))

                for t in count():
                    if M_e.is_full:
                        if forget_policy == "random":
                            M_e.forget_random()

                        elif forget_policy == "generalize":

                            mem_epi = M_e.find_mem_for_semantic()
                            M_e.forget(mem_epi)
                            mem_sem = SemanticMemory.ob2sem(mem_epi)
                            M_s.add(mem_sem)

                            if M_s.is_full:
                                M_s.forget_weakest()

                        else:
                            raise ValueError

                    if M_s.is_full:
                        assert forget_policy == "random"
                        M_s.forget_random()

                    if answer_policy == "episem":
                        if M_e.is_answerable(question):
                            pred, _ = M_e.answer_latest(question)
                        else:
                            pred, _ = M_s.answer_strongest(question)
                    else:
                        pred = random.choice(
                            [M_e.answer_random(question), M_s.answer_random]
                        )

                    (ob, question), reward, done, info = env.step(pred)
                    ob = ob[0]

                    if forget_policy == "random":
                        if random.random() < 0.5:
                            M_e.add(EpisodicMemory.ob2epi(ob))
                        else:
                            M_s.add(SemanticMemory.ob2sem(ob))
                    else:
                        M_e.add(EpisodicMemory.ob2epi(ob))

                    rewards[f"{forget_policy}_{answer_policy}"] += reward

                    if done:
                        break

        rewards["info"] = {"seed": seed, "capacity": capacity}
        results.append(rewards)

    return results


def episodic_semantic_pretrain(env_params, caps, seed):

    seed_everything(seed)
    env = RoomEnv(**env_params)

    results = []
    for capacity in caps:
        rewards = {}
        for forget_policy in ["oldest", "random"]:
            for answer_policy in ["episem", "random"]:

                rewards[f"{forget_policy}_{answer_policy}"] = 0
                M_e = EpisodicMemory(capacity // 2)
                M_s = SemanticMemory(capacity // 2)

                free_space = M_s.pretrain_semantic(env)
                M_e.increase_capacity(free_space)
                assert M_e.capacity + M_s.capacity == capacity

                ob, question = env.reset()
                ob = ob[0]
                M_e.add(EpisodicMemory.ob2epi(ob))

                for t in count():
                    if M_e.is_full:
                        if forget_policy == "random":
                            M_e.forget_random()

                        elif forget_policy == "oldest":

                            M_e.forget_oldest()
                        else:
                            raise ValueError

                    if answer_policy == "episem":
                        if M_e.is_answerable(question):
                            pred, _ = M_e.answer_latest(question)
                        else:
                            pred, _ = M_s.answer_strongest(question)
                    else:
                        pred = random.choice(
                            [M_e.answer_random(question), M_s.answer_random]
                        )

                    (ob, question), reward, done, info = env.step(pred)
                    ob = ob[0]

                    M_e.add(EpisodicMemory.ob2epi(ob))

                    rewards[f"{forget_policy}_{answer_policy}"] += reward

                    if done:
                        break

        rewards["info"] = {"seed": seed, "capacity": capacity}
        results.append(rewards)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--caps", type=int, required=True)
    args = parser.parse_args()

    caps = [args.caps]
    to_dump = {}
    for common_sense in tqdm([0.1, 0.3, 0.5, 0.7]):
        for new_location in [0.1, 0.3, 0.5, 0.7]:
            for new_object in [0.1, 0.3, 0.5, 0.7]:
                for switch_person in [0.1, 0.3, 0.5, 0.7]:
                    env_params = {
                        "semantic_knowledge_path": "./data/semantic-knowledge-small.json",
                        "names_path": "./data/top-human-names-small",
                        "weighting_mode": "highest",
                        "probs": {
                            "commonsense": common_sense,
                            "new_location": new_location,
                            "new_object": new_object,
                            "switch_person": switch_person,
                        },
                        "limits": {
                            "heads": None,
                            "tails": None,
                            "names": None,
                            "allow_spaces": False,
                        },
                        "max_step": 100,
                        "disjoint_entities": True,
                        "num_agents": 1,
                    }
                    seed_everything(42)
                    env = RoomEnv(**env_params)

                    results_all = {}
                    results_all["episodic"] = []
                    results_all["semantic"] = []
                    results_all["both"] = []
                    results_all["both-presem"] = []
                    seeds = [0, 1, 2]

                    for seed in seeds:
                        results_all["episodic"].append(episodic(env_params, caps, seed))
                        results_all["semantic"].append(semantic(env_params, caps, seed))
                        results_all["both"].append(
                            episodic_semantic(env_params, caps, seed)
                        )
                        results_all["both-presem"].append(
                            episodic_semantic_pretrain(env_params, caps, seed)
                        )

                    target = []
                    others = []

                    for key, val in results_all.items():
                        # print(key)
                        for val_ in val:
                            for val__ in val_:
                                for k, v in val__.items():
                                    if k == "info":
                                        continue
                                    # print(k, v)
                                    if "random" in k:
                                        others.append(v)
                                    else:
                                        target.append(v)

                    diff = np.average(target) - np.average(others)
                    diff = diff.item()

                    to_dump[
                        f"{common_sense}_{new_location}_{new_object}_{switch_person}"
                    ] = diff

    write_json(to_dump, f"{args.caps}.json")
