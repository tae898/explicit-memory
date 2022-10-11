"""Matplotlib functions."""
import logging
import os
from copy import deepcopy
from glob import glob
from typing import List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import room_env
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from train import DQNLightning, RLAgent
from utils import read_yaml

logger = logging.getLogger()
logger.disabled = True


def run_seed(
    seed: int,
    agent_config: dict,
    des_size: str,
    capacity: int,
    allow_random_human: bool,
    allow_random_question: bool,
    episodic_replace_old: bool,
    semantic_replace_old: bool,
) -> int:
    """Run with seed."""

    env = gym.make(
        "RoomEnv-v1",
        des_size=des_size,
        seed=seed,
        allow_random_human=allow_random_human,
        allow_random_question=allow_random_question,
    )

    agent_config["seed"] = seed
    agent_config["env"] = env
    agent_config["episodic_replace_old"] = episodic_replace_old
    agent_config["semantic_replace_old"] = semantic_replace_old

    if agent_config["forget_policy"]["episodic"] is None:
        agent_config["capacity"] = {
            "episodic": 0,
            "semantic": capacity,
            "short": 1,
        }

    if agent_config["forget_policy"]["semantic"] is None:
        agent_config["capacity"] = {
            "episodic": capacity,
            "semantic": 0,
            "short": 1,
        }

    if (agent_config["forget_policy"]["episodic"] is not None) and (
        agent_config["forget_policy"]["semantic"] is not None
    ):
        agent_config["capacity"] = {
            "episodic": capacity // 2,
            "semantic": capacity // 2,
            "short": 1,
        }

    agent = HandcraftedAgent(**agent_config)
    agent.run()
    rewards = agent.rewards

    return rewards


def load_episodic_semantic_random_scratch_pretrained(
    data_dir: str = "./data/",
    kind: str = "test_total_reward_mean",
    capacity: int = 32,
    des_size: str = "l",
) -> dict:
    results = {}

    results["scratch"] = load_training_val_test_results(
        data_dir=data_dir,
        kind=kind,
        capacity=capacity,
        pretrain=False,
        des_size=des_size,
    )
    results["pretrained"] = load_training_val_test_results(
        data_dir=data_dir,
        kind=kind,
        capacity=capacity,
        pretrain=True,
        des_size=des_size,
    )

    results["random"] = run_seeds(
        agent_config={
            "forget_policy": {
                "episodic": "oldest",
                "semantic": "weakest",
                "short": "random",
            },
            "answer_policy": {
                "episodic": "latest",
                "semantic": "strongest",
            },
            "pretrain_semantic": False,
        },
        des_size=des_size,
        capacity=capacity,
        allow_random_human=False,
        allow_random_question=False,
        episodic_replace_old=False,
        semantic_replace_old=False,
        seeds=[0, 1, 2, 3, 4],
    )

    results["episodic"] = run_seeds(
        agent_config={
            "forget_policy": {
                "episodic": "oldest",
                "semantic": None,
                "short": "episodic",
            },
            "answer_policy": {
                "episodic": "latest",
                "semantic": None,
            },
            "pretrain_semantic": False,
        },
        des_size=des_size,
        capacity=capacity,
        allow_random_human=False,
        allow_random_question=False,
        episodic_replace_old=False,
        semantic_replace_old=False,
        seeds=[0, 1, 2, 3, 4],
    )

    results["semantic"] = run_seeds(
        agent_config={
            "forget_policy": {
                "episodic": None,
                "semantic": "weakest",
                "short": "semantic",
            },
            "answer_policy": {
                "episodic": None,
                "semantic": "strongest",
            },
            "pretrain_semantic": False,
        },
        des_size=des_size,
        capacity=capacity,
        allow_random_human=False,
        allow_random_question=False,
        episodic_replace_old=False,
        semantic_replace_old=False,
        seeds=[0, 1, 2, 3, 4],
    )

    return results


def run_seeds(
    agent_config: dict,
    des_size: str,
    capacity: int,
    allow_random_human: bool,
    allow_random_question: bool,
    episodic_replace_old: bool,
    semantic_replace_old: bool,
    seeds: list,
) -> dict:

    rewards = []

    for seed in seeds:
        rewards_ = run_seed(
            seed=seed,
            agent_config=agent_config,
            des_size=des_size,
            capacity=capacity,
            allow_random_human=allow_random_human,
            allow_random_question=allow_random_question,
            episodic_replace_old=episodic_replace_old,
            semantic_replace_old=semantic_replace_old,
        )
        rewards.append(rewards_)

    mean_rewards, std_rewards = np.mean(rewards), np.std(rewards)

    return {"mean": mean_rewards, "std": std_rewards}


def run_all_agent_configs(
    des_size: str = "l",
    capacity: int = 32,
    allow_random_human: bool = False,
    allow_random_question: bool = False,
    episodic_replace_old: bool = False,
    semantic_replace_old: bool = False,
    seeds: list = [0, 1, 2, 3, 4],
    agents: list = [
        "episodic",
        "semantic",
        "random",
        "scratch",
        "pretrained",
    ],
) -> List:
    agent_configs = []

    # Only episodic
    if "episodic" in agents:
        for forget_policy_episodic in ["oldest", "random"]:
            for answer_policy_episodic in ["latest", "random"]:
                agent_configs.append(
                    {
                        "forget_policy": {
                            "episodic": forget_policy_episodic,
                            "semantic": None,
                            "short": "episodic",
                        },
                        "answer_policy": {
                            "episodic": answer_policy_episodic,
                            "semantic": None,
                        },
                        "pretrain_semantic": False,
                    }
                )

    # only semantic
    if "semantic" in agents:
        for forget_policy_semantic in ["weakest", "random"]:
            for answer_policy_semantic in ["strongest", "random"]:
                agent_configs.append(
                    {
                        "forget_policy": {
                            "episodic": None,
                            "semantic": forget_policy_semantic,
                            "short": "semantic",
                        },
                        "answer_policy": {
                            "episodic": None,
                            "semantic": answer_policy_semantic,
                        },
                        "pretrain_semantic": False,
                    }
                )

    # both
    if "random" in agents:
        for forget_policy_short in ["generalize", "random"]:
            for answer_policy_episodic in ["latest", "random"]:
                for answer_policy_semantic in ["strongest", "random"]:
                    agent_configs.append(
                        {
                            "forget_policy": {
                                "episodic": "oldest",
                                "semantic": "weakest",
                                "short": forget_policy_short,
                            },
                            "answer_policy": {
                                "episodic": answer_policy_episodic,
                                "semantic": answer_policy_semantic,
                            },
                            "pretrain_semantic": False,
                        }
                    )

    # both presem
    for answer_policy_episodic in ["latest", "random"]:
        for answer_policy_semantic in ["strongest", "random"]:
            agent_configs.append(
                {
                    "forget_policy": {
                        "episodic": "oldest",
                        "semantic": "weakest",
                        "short": "generalize",
                    },
                    "answer_policy": {
                        "episodic": answer_policy_episodic,
                        "semantic": answer_policy_semantic,
                    },
                    "pretrain_semantic": True,
                }
            )

    rewards_by_config = []
    for agent_config in agent_configs:
        mean_rewards = run_seeds(
            agent_config=deepcopy(agent_config),
            des_size=des_size,
            capacity=capacity,
            allow_random_human=allow_random_human,
            allow_random_question=allow_random_question,
            episodic_replace_old=episodic_replace_old,
            semantic_replace_old=semantic_replace_old,
            seeds=seeds,
        )
        rewards_by_config.append(
            {"mean_rewards": mean_rewards, "agent_config": deepcopy(agent_config)}
        )

    return rewards_by_config


def run_all_capacities(
    des_size: str = "l",
    allow_random_human: bool = False,
    allow_random_question: bool = False,
    episodic_replace_old: bool = False,
    semantic_replace_old: bool = False,
    seeds: list = [0, 1, 2, 3, 4],
    agents: list = [
        "episodic",
        "semantic",
        "random",
        "scratch",
        "pretrained",
    ],
) -> dict:
    rewards_by_capacity = {}
    for capacity in [2, 4, 8, 16, 32, 64]:
        rewards_by_config = run_all_agent_configs(
            des_size=des_size,
            capacity=capacity,
            allow_random_human=allow_random_human,
            allow_random_question=allow_random_question,
            episodic_replace_old=episodic_replace_old,
            semantic_replace_old=semantic_replace_old,
            seeds=seeds,
            agents=agents,
        )
        rewards_by_capacity[capacity] = rewards_by_config

    return rewards_by_capacity


def run_all_sizes(
    allow_random_human: bool = False,
    allow_random_question: bool = False,
    episodic_replace_old: bool = False,
    semantic_replace_old: bool = False,
    des_sizes: list = ["l"],
    seeds: list = [0, 1, 2, 3, 4],
    agents: list = [
        "episodic",
        "semantic",
        "random",
        "scratch",
        "pretrained",
    ],
) -> dict:
    rewards_by_size = {}
    for des_size in tqdm(des_sizes):
        rewards_by_capacity = run_all_capacities(
            des_size=des_size,
            allow_random_human=allow_random_human,
            allow_random_question=allow_random_question,
            episodic_replace_old=episodic_replace_old,
            semantic_replace_old=semantic_replace_old,
            seeds=seeds,
            agents=agents,
        )
        rewards_by_size[des_size] = rewards_by_capacity

    return rewards_by_size


def plot_by_des_hand_crafted_only(
    des_size, results, save_dir: str = "./figures/"
) -> None:
    results_ = results[des_size]

    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.asanyarray([i for i in range(len(results_))])
    width = 0.2

    legend_order = [
        "Hand-crafted 1: Only episodic",
        "Hand-crafted 2: Only semantic",
        "Hand-crafted 3: Both, random",
        "Hand-crafted 4: Both, pre-sem",
    ]

    color_order = ["orange", "dodgerblue", "yellowgreen", "deeppink"]

    stats = {"episodic": {}, "semantic": {}, "both_random": {}, "both_presem": {}}
    for capacity, rewards in results_.items():
        for rewards_ in rewards:
            if rewards_["agent_config"] == {
                "forget_policy": {
                    "episodic": "oldest",
                    "semantic": None,
                    "short": "episodic",
                },
                "answer_policy": {"episodic": "latest", "semantic": None},
                "pretrain_semantic": False,
            }:
                stats["episodic"][capacity] = rewards_

            if rewards_["agent_config"] == {
                "forget_policy": {
                    "episodic": None,
                    "semantic": "weakest",
                    "short": "semantic",
                },
                "answer_policy": {"episodic": None, "semantic": "strongest"},
                "pretrain_semantic": False,
            }:
                stats["semantic"][capacity] = rewards_

            if rewards_["agent_config"] == {
                "forget_policy": {
                    "episodic": "oldest",
                    "semantic": "weakest",
                    "short": "random",
                },
                "answer_policy": {"episodic": "latest", "semantic": "strongest"},
                "pretrain_semantic": False,
            }:
                stats["both_random"][capacity] = rewards_

            if rewards_["agent_config"] == {
                "forget_policy": {
                    "episodic": "oldest",
                    "semantic": "weakest",
                    "short": "generalize",
                },
                "answer_policy": {"episodic": "latest", "semantic": "strongest"},
                "pretrain_semantic": True,
            }:
                stats["both_presem"][capacity] = rewards_

    agent_types = ["episodic", "semantic", "both_random", "both_presem"]
    for agent_type, w, color in zip(agent_types, [-1.5, -0.5, 0.5, 1.5], color_order):
        caps = [key for key in stats[agent_type]]
        assert sorted(caps) == caps
        means = [val["mean_rewards"]["mean"] for key, val in stats[agent_type].items()]
        stds = [val["mean_rewards"]["std"] for key, val in stats[agent_type].items()]

        print(agent_type, means)

        ax.bar(
            x=idx + w * width,
            height=means,
            yerr=stds,
            width=width,
            color=color,
            capsize=4,
        )
        ax.set_xticks(idx)
        ax.set_xticklabels(list(results_.keys()))
        ax.legend(legend_order, fontsize=10, loc="upper left")
        ax.set_xlabel("memory capacity", fontsize=15)
        ax.set_ylabel("total average rewards", fontsize=15)
        ax.set_ylim([0, 128])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.title(f"Best performance by memory type, des_size={des_size}", fontsize=18)
    plt.savefig(os.path.join(save_dir, f"des-size_{des_size}_best-strategies.pdf"))


def plot_by_des2(des_size, results, save_dir: str = "./figures/") -> None:
    results_ = results[des_size]

    stats = {}
    for agent_type in ["episodic", "semantic", "both_random", "both_presem"]:

        if agent_type == "episodic":
            stats[agent_type] = {}
            legend_order = [
                "forget oldest, answer latest",
                "forget random, answer latest",
                "forget oldest, answer random",
                "forget random, answer random",
            ]
            stats[agent_type] = {foo: {} for foo in legend_order}
            color_order = ["orange", "navajowhite", "blanchedalmond", "oldlace"]

            for capacity, rewards in results_.items():
                for rewards_ in rewards:
                    if rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": None,
                            "short": "episodic",
                        },
                        "answer_policy": {"episodic": "latest", "semantic": None},
                        "pretrain_semantic": False,
                    }:
                        stats["episodic"]["forget oldest, answer latest"][
                            capacity
                        ] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "random",
                            "semantic": None,
                            "short": "episodic",
                        },
                        "answer_policy": {"episodic": "latest", "semantic": None},
                        "pretrain_semantic": False,
                    }:
                        stats["episodic"]["forget random, answer latest"][
                            capacity
                        ] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": None,
                            "short": "episodic",
                        },
                        "answer_policy": {"episodic": "random", "semantic": None},
                        "pretrain_semantic": False,
                    }:
                        stats["episodic"]["forget oldest, answer random"][
                            capacity
                        ] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "random",
                            "semantic": None,
                            "short": "episodic",
                        },
                        "answer_policy": {"episodic": "random", "semantic": None},
                        "pretrain_semantic": False,
                    }:
                        stats["episodic"]["forget random, answer random"][
                            capacity
                        ] = rewards_

        elif agent_type == "semantic":
            stats[agent_type] = {}
            legend_order = [
                "forget weakest, answer strongest",
                "forget random, answer strongest",
                "forget weakest, answer random",
                "forget random, answer random",
            ]
            stats[agent_type] = {foo: {} for foo in legend_order}
            color_order = ["dodgerblue", "lightskyblue", "powderblue", "aliceblue"]

            for capacity, rewards in results_.items():
                for rewards_ in rewards:
                    if rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": None,
                            "semantic": "weakest",
                            "short": "semantic",
                        },
                        "answer_policy": {"episodic": None, "semantic": "strongest"},
                        "pretrain_semantic": False,
                    }:
                        stats["semantic"]["forget weakest, answer strongest"][
                            capacity
                        ] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": None,
                            "semantic": "random",
                            "short": "semantic",
                        },
                        "answer_policy": {"episodic": None, "semantic": "strongest"},
                        "pretrain_semantic": False,
                    }:
                        stats["semantic"]["forget random, answer strongest"][
                            capacity
                        ] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": None,
                            "semantic": "weakest",
                            "short": "semantic",
                        },
                        "answer_policy": {"episodic": None, "semantic": "random"},
                        "pretrain_semantic": False,
                    }:
                        stats["semantic"]["forget weakest, answer random"][
                            capacity
                        ] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": None,
                            "semantic": "random",
                            "short": "semantic",
                        },
                        "answer_policy": {"episodic": None, "semantic": "random"},
                        "pretrain_semantic": False,
                    }:
                        stats["semantic"]["forget random, answer random"][
                            capacity
                        ] = rewards_

        elif agent_type == "both_random":
            stats[agent_type] = {}
            legend_order = [
                "answer episodic latest, answer semantic strongest",
                "answer episodic random, answer semantic strongest",
                "answer episodic latest, answer semantic random",
                "answer episodic random, answer semantic random",
            ]
            stats[agent_type] = {foo: {} for foo in legend_order}
            color_order = ["yellowgreen", "lightgreen", "palegreen", "honeydew"]

            for capacity, rewards in results_.items():
                for rewards_ in rewards:
                    if rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": "weakest",
                            "short": "random",
                        },
                        "answer_policy": {
                            "episodic": "latest",
                            "semantic": "strongest",
                        },
                        "pretrain_semantic": False,
                    }:
                        stats["both_random"][
                            "answer episodic latest, answer semantic strongest"
                        ][capacity] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": "weakest",
                            "short": "random",
                        },
                        "answer_policy": {
                            "episodic": "random",
                            "semantic": "strongest",
                        },
                        "pretrain_semantic": False,
                    }:
                        stats["both_random"][
                            "answer episodic random, answer semantic strongest"
                        ][capacity] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": "weakest",
                            "short": "random",
                        },
                        "answer_policy": {"episodic": "latest", "semantic": "random"},
                        "pretrain_semantic": False,
                    }:
                        stats["both_random"][
                            "answer episodic latest, answer semantic random"
                        ][capacity] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": "weakest",
                            "short": "random",
                        },
                        "answer_policy": {"episodic": "random", "semantic": "random"},
                        "pretrain_semantic": False,
                    }:
                        stats["both_random"][
                            "answer episodic random, answer semantic random"
                        ][capacity] = rewards_

        elif agent_type == "both_presem":
            stats[agent_type] = {}
            legend_order = [
                "answer episodic latest, answer semantic strongest",
                "answer episodic random, answer semantic strongest",
                "answer episodic latest, answer semantic random",
                "answer episodic random, answer semantic random",
            ]
            stats[agent_type] = {foo: {} for foo in legend_order}
            color_order = ["deeppink", "lightpink", "pink", "lavenderblush"]

            for capacity, rewards in results_.items():
                for rewards_ in rewards:
                    if rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": "weakest",
                            "short": "generalize",
                        },
                        "answer_policy": {
                            "episodic": "latest",
                            "semantic": "strongest",
                        },
                        "pretrain_semantic": True,
                    }:
                        stats["both_presem"][
                            "answer episodic latest, answer semantic strongest"
                        ][capacity] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": "weakest",
                            "short": "generalize",
                        },
                        "answer_policy": {
                            "episodic": "random",
                            "semantic": "strongest",
                        },
                        "pretrain_semantic": True,
                    }:
                        stats["both_presem"][
                            "answer episodic random, answer semantic strongest"
                        ][capacity] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": "weakest",
                            "short": "generalize",
                        },
                        "answer_policy": {"episodic": "latest", "semantic": "random"},
                        "pretrain_semantic": True,
                    }:
                        stats["both_presem"][
                            "answer episodic latest, answer semantic random"
                        ][capacity] = rewards_

                    elif rewards_["agent_config"] == {
                        "forget_policy": {
                            "episodic": "oldest",
                            "semantic": "weakest",
                            "short": "generalize",
                        },
                        "answer_policy": {"episodic": "random", "semantic": "random"},
                        "pretrain_semantic": True,
                    }:
                        stats["both_presem"][
                            "answer episodic random, answer semantic random"
                        ][capacity] = rewards_

        else:
            raise ValueError

        fig, ax = plt.subplots(figsize=(10, 5))
        for legend_, w, color in zip(legend_order, [-1.5, -0.5, 0.5, 1.5], color_order):
            caps = [key for key in stats[agent_type][legend_]]

            idx = np.asanyarray([i for i in range(len(caps))])
            width = 0.2
            assert sorted(caps) == caps

            means = [
                val["mean_rewards"]["mean"]
                for val in stats[agent_type][legend_].values()
            ]

            stds = [
                val["mean_rewards"]["std"]
                for val in stats[agent_type][legend_].values()
            ]

            print(agent_type, legend_, means)
            ax.bar(
                x=idx + w * width,
                height=means,
                yerr=stds,
                width=width,
                color=color,
                capsize=4,
            )
            ax.set_xticks(idx)
            ax.set_xticklabels(caps)
            ax.legend(legend_order, fontsize=10, loc="upper left")
            ax.set_xlabel("memory capacity", fontsize=15)
            ax.set_ylabel(f"total rewards", fontsize=15)

            ax.set_ylim([0, 128])

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title(
            f"Performance by different forget / answer rules, "
            f"agent_type={agent_type}, des_size={des_size}",
            fontsize=18,
        )
        plt.savefig(
            os.path.join(save_dir, f"des_size_{des_size}_agent-type_{agent_type}.pdf")
        )


def load_training_val_test_results(
    data_dir: str = "./data/",
    kind: str = "train_total_reward",
    capacity: int = 32,
    pretrain: bool = False,
    des_size: str = "l",
) -> np.ndarray:
    assert kind in [
        "train_loss",
        "train_total_reward",
        "val_total_reward_mean",
        "test_total_reward_mean",
    ]
    paths = glob(
        os.path.join(
            data_dir,
            kind,
            "*.csv",
        )
    )
    paths = [
        path
        for path in paths
        if f"capacity={capacity}" in path
        and (
            (f"pretrain_semantic={pretrain}" in path)
            or (f"pretrain={pretrain}" in path)
        )
        and f"des_size={des_size}" in path
    ]
    assert len(paths) == 5
    runs = []
    for path in paths:
        df = pd.read_csv(path)
        values = df.Value.to_numpy()
        if "train" in kind or "val" in kind:
            if "reward" in kind:
                assert len(values) == 16
            else:
                assert len(values) == 1000
        elif "test" in kind:
            assert len(values) == 1
        else:
            raise ValueError
        runs.append(values)

    runs = np.array(runs)
    means = np.mean(runs, axis=0)
    stds = np.std(runs, axis=0)
    steps = df.Step.to_numpy()

    assert len(means) == len(stds) == len(steps)

    if len(means) == 1:
        return {"mean": means.item(), "std": stds.item(), "step": steps.item()}
    else:
        return {"means": means, "stds": stds, "steps": steps}


def plot_training_validation_results(
    data_dir: str = "./data/",
    kind: str = "train_total_reward",
    capacity: int = 32,
    save_dir: str = "./figures/",
    ymin: int = 64,
    ymax: int = 128,
    ylog: bool = False,
    xlabel: str = "Step",
    des_size: str = "l",
    figsize: Tuple = (10, 10),
    legend_loc: str = "upper left",
) -> None:
    assert kind in [
        "train_loss",
        "train_total_reward",
        "val_total_reward_mean",
    ]

    if kind == "train_loss":
        title = "Avg. loss, training."
        ylabel = "Avg. loss"

    elif kind == "train_total_reward":
        title = "Avg. total rewards, training."
        ylabel = "Avg. total rewards"

    elif kind == "val_total_reward_mean":
        title = "Avg. total rewards, validation."
        ylabel = "Avg. total rewards"

    else:
        raise ValueError

    fig, ax = plt.subplots(figsize=figsize)

    training_val_test_results = load_training_val_test_results(
        data_dir=data_dir,
        kind=kind,
        capacity=capacity,
        pretrain=False,
        des_size=des_size,
    )
    means, stds, steps = (
        training_val_test_results["means"],
        training_val_test_results["stds"],
        training_val_test_results["steps"],
    )
    print(kind, "pretrain=False", means, stds, steps)
    ax.plot(steps, means, color="pink")
    ax.fill_between(
        steps,
        means - stds,
        means + stds,
        alpha=0.2,
        edgecolor="pink",
        facecolor="pink",
        label="_nolegend_",
    )

    training_val_test_results = load_training_val_test_results(
        data_dir=data_dir,
        kind=kind,
        capacity=capacity,
        pretrain=True,
        des_size=des_size,
    )
    means, stds, steps = (
        training_val_test_results["means"],
        training_val_test_results["stds"],
        training_val_test_results["steps"],
    )
    print(kind, "pretrain=False", f"means: {means}")
    ax.plot(steps, means, color="deeppink")
    ax.fill_between(
        steps,
        means - stds,
        means + stds,
        alpha=0.2,
        edgecolor="deeppink",
        facecolor="deeppink",
        label="_nolegend_",
    )
    if ymin is None and ymax is None:
        ymin, ymax = ax.get_ylim()

    ax.set_ylim(ymin, ymax)
    # ax.yaxis.set_ticks(np.arange(ymin, ymax, 7))

    if ylog:
        plt.yscale("log")
        ylabel += " (log scale)"
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=30)
    ax.set_xlabel(xlabel, fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    ax.legend(
        ["Semantic memory from scratch", "Semantic memory prefilled"],
        fontsize=25,
        loc=legend_loc,
    )

    plt.title(title, fontsize=40)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if "v1" in data_dir:
        filename = f"des_size={des_size}-capacity={capacity}-{kind}-v1.pdf"
    elif "v2" in data_dir:
        filename = f"des_size={des_size}-capacity={capacity}-{kind}-v2.pdf"
    else:
        filename = f"des_size={des_size}-capacity={capacity}-{kind}.pdf"

    plt.savefig(os.path.join(save_dir, filename))


def plot_test_results(
    data_dir: str = "./data/",
    capacity: int = 32,
    save_dir: str = "./figures/",
    ymin: int = 64,
    ymax: int = 128,
    des_size: str = "l",
    figsize: Tuple = (10, 10),
    legend_loc: str = "upper left",
) -> None:
    results = load_episodic_semantic_random_scratch_pretrained(
        data_dir=data_dir,
        kind="test_total_reward_mean",
        capacity=capacity,
        des_size=des_size,
    )

    fig, ax = plt.subplots(figsize=figsize)
    title = "Avg. total rewards, test."
    ylabel = "Avg. total rewards"

    legend_order = [
        "Episodic only, handcrafted",
        "Semantic only, handcrafted",
        "Both, random",
        "Both, RL, semantic from scratch",
        "Both, RL, semantic pretrained",
    ]
    color_order = ["orange", "dodgerblue", "gray", "pink", "deeppink"]
    agent_order = ["episodic", "semantic", "random", "scratch", "pretrained"]

    for legend, color, agent in zip(legend_order, color_order, agent_order):
        height = results[agent]["mean"]
        yerr = results[agent]["std"]
        ax.bar(
            x=legend,
            height=height,
            color=color,
            width=0.9,
            yerr=yerr,
            capsize=4,
        )
        print(f"heights: {height}, stds: {yerr}")
    plt.xticks([])
    ax.set_ylim([ymin, ymax])
    plt.yticks(fontsize=30)
    # ax.yaxis.set_ticks(np.arange(ymin, ymax, 7))

    ax.legend(legend_order, fontsize=20, loc=legend_loc)
    ax.set_xlabel("Agent type", fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    plt.title(title, fontsize=40)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        os.path.join(
            save_dir,
            f"des_size={des_size}-capacity={capacity}-test_total_reward_mean.pdf",
        )
    )


def plot_test_results_all_capacities(
    data_dir: str = "./data/",
    capacity: int = 32,
    save_dir: str = "./figures/",
    ymin: int = 0,
    ymax: int = 128,
    des_size: str = "l",
    figsize: Tuple = (25, 8),
    legend_loc: str = "upper left",
) -> None:

    capacities = [2, 4, 8, 16, 32, 64]
    width = 0.17

    results = {}
    for capacity in capacities:
        results_ = load_episodic_semantic_random_scratch_pretrained(
            data_dir=data_dir,
            kind="test_total_reward_mean",
            capacity=capacity,
            des_size=des_size,
        )
        results[capacity] = results_

    idx = np.asanyarray([i for i in range(len(results))])

    fig, ax = plt.subplots(figsize=figsize)
    title = "Avg. total rewards, varying capacities, test."
    ylabel = "Avg. total rewards"

    legend_order = [
        "Episodic only, handcrafted",
        "Semantic only, handcrafted",
        "Both, random",
        "Both, RL, semantic from scratch",
        "Both, RL, semantic pretrained",
    ]

    color_order = ["orange", "dodgerblue", "gray", "pink", "deeppink"]
    agent_order = ["episodic", "semantic", "random", "scratch", "pretrained"]

    for w, color, agent in zip([-2, -1, 0, 1, 2], color_order, agent_order):
        height = [results_[agent]["mean"] for _, results_ in results.items()]
        yerr = [results_[agent]["std"] for _, results_ in results.items()]
        ax.bar(
            x=idx + w * width,
            height=height,
            yerr=yerr,
            width=width,
            color=color,
            capsize=4,
        )
        print(f"agent: {agent}, height: {height}, std: {yerr}")
    ax.set_xticks(idx)
    ax.set_xticklabels(list(results.keys()))
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=30)
    ax.set_ylim([ymin, ymax])
    ax.legend(legend_order, fontsize=30, loc=legend_loc)
    ax.set_xlabel("Memory capacity", fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40)
    plt.title(title, fontsize=50)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        os.path.join(
            save_dir,
            f"des_size={des_size}-capacity=all-test_total_reward_mean.pdf",
        )
    )


class UnderstandModel:
    def __init__(
        self,
        model_scratch_path: str = "./models/des_size=l-capacity=32-pretrain=False-gpus=0-seed=1/checkpoints/epoch=07-val_total_reward_mean=115.50-val_total_reward_std=3.14.ckpt",
        model_pretrained_path: str = "./models/des_size=l-capacity=32-pretrain=True-gpus=0-seed=2/checkpoints/epoch=10-val_total_reward_mean=121.00-val_total_reward_std=1.55.ckpt",
    ) -> None:
        """Call the model loading."""
        self.model_scratch_path = model_scratch_path
        self.model_pretrained_path = model_pretrained_path

        self._load_models()

        self.indices = {
            interest: {
                i: key
                for i, (key, val) in enumerate(
                    self.embeddings["scratch"][interest].items()
                )
            }
            for interest in ["humans", "objects", "object_locations"]
        }

    def _load_models(self) -> None:
        """Load the models and its embeddings.

        I don't know why but this takes some time.
        """
        self.models = {}
        self.embeddings = {}
        for semantic in ["scratch", "pretrained"]:
            if semantic == "scratch":
                model_path = self.model_scratch_path
            else:
                model_path = self.model_pretrained_path

            self.models[semantic] = DQNLightning.load_from_checkpoint(model_path)
            self.models[semantic].eval()

            self.word2idx = self.models[semantic].net.word2idx

            self.embeddings[semantic] = {}

            self.embeddings[semantic]["humans"] = {
                human: self.models[semantic]
                .net.embeddings(torch.tensor(self.word2idx[human]))
                .detach()
                .cpu()
                .numpy()
                for human in self.models[semantic].env.des.humans
            }

            self.embeddings[semantic]["objects"] = {
                obj: self.models[semantic]
                .net.embeddings(torch.tensor(self.word2idx[obj]))
                .detach()
                .cpu()
                .numpy()
                for obj in self.models[semantic].env.des.objects
            }

            self.embeddings[semantic]["object_locations"] = {
                obj_loc: self.models[semantic]
                .net.embeddings(torch.tensor(self.word2idx[obj_loc]))
                .detach()
                .cpu()
                .numpy()
                for obj_loc in self.models[semantic].env.des.object_locations
            }

    def compute_reduction(
        self,
        tsne_params: dict = None,
        pca_params: dict = None,
    ) -> None:
        """Compute dimension reduction on the embeddings.

        Args
        ----
        tsne_params: t-SNE parameters, e.g., {"n_components": 2, "perplexity": 3}
            This is a non-linear reduction method. See the paper.
        pca_params: PCA parameters, e.g., {"n_components": 2}
            This is a linear reduction method. See wikipedia.

        """
        if tsne_params is None:
            assert pca_params is not None
            self.reduction_method = "PCA"
        if pca_params is None:
            assert tsne_params is not None
            self.reduction_method = "t-SNE"

        self.X = {}
        for semantic in ["scratch", "pretrained"]:
            self.X[semantic] = {}
            for interest in ["humans", "objects", "object_locations"]:
                X = np.array(list(self.embeddings[semantic][interest].values()))

                if self.reduction_method == "t-SNE":
                    self.X[semantic][interest] = TSNE(**tsne_params).fit_transform(X)
                else:
                    pca = PCA(**pca_params)
                    pca.fit(X)
                    self.X[semantic][interest] = pca.transform(X)

    def plot_embeddings(
        self,
        semantic: str = "scratch",
        interest: str = "humans",
        figsize: Tuple = (10, 10),
        save_dir: str = "./figures/",
    ) -> None:
        self.semantic = semantic
        self.interest = interest
        if semantic == "scratch":
            color = "pink"
        else:
            color = "deeppink"
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            self.X[self.semantic][self.interest][:, 0],
            self.X[self.semantic][self.interest][:, 1],
            color=color,
            edgecolors="black",
            s=250,
        )
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        ax.set_xlabel(f"{self.reduction_method} dimension0", fontsize=35)
        ax.set_ylabel(f"{self.reduction_method} dimension1", fontsize=35)

        plt.title(
            f"{self.interest} embeddings\n" f"semantic-{semantic}",
            fontsize=40,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            os.path.join(
                save_dir, f"{self.reduction_method}-{self.semantic}-{self.interest}.pdf"
            )
        )

    def get_similar_embeddings(
        self,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
    ) -> List:
        condition_xmin = xmin < self.X[self.semantic][self.interest][:, 0]
        condition_xmax = xmax > self.X[self.semantic][self.interest][:, 0]
        condition_ymin = ymin < self.X[self.semantic][self.interest][:, 1]
        condition_ymax = ymax > self.X[self.semantic][self.interest][:, 1]

        condition = condition_xmin * condition_xmax * condition_ymin * condition_ymax
        idx_embs = np.where(condition)[0]

        entities = [self.indices[self.interest][idx] for idx in idx_embs]

        return entities


def rename_test_debug_results():
    """Rename the lightning log directories for easier access."""
    test_debug_paths = glob("./lightning_logs/*/test_debug*")

    for test_debug_path in tqdm(test_debug_paths):
        hparams_path = os.path.join(*test_debug_path.split("/")[:-1], "hparams.yaml")
        hparams = read_yaml(hparams_path)
        des_size = hparams["des_size"]
        capacity = hparams["capacity"]["episodic"] + hparams["capacity"]["semantic"]
        # loc = hparams["nn_params"]["human_embedding_on_object_location"]
        pretrain = hparams["pretrain_semantic"]
        seed = hparams["seed"]
        gpus = hparams["gpus"]
        new_dir_name = f"des_size={des_size}-capacity={capacity}-pretrain={pretrain}-gpus={gpus}-seed={seed}"
        new_dir_name = os.path.join(*test_debug_path.split("/")[:-2], new_dir_name)
        old_dir_name = os.path.join(*test_debug_path.split("/")[:-1])
        os.rename(old_dir_name, new_dir_name)
