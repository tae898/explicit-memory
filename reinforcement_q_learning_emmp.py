# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger()
logger.disabled = True
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from memory.environment.gym import MemoryEnv
from memory.utils import seed_everything
from memory.model import create_policy_net
from memory import EpisodicMemory

config = {
    "seed": 1,
    "training_params": {
        "algorithm": "policy_gradients",
        "device": "cpu",
        "precision": 32,
        "num_processes": 16,
        "gamma": 0,
        "learning_rate": 0.001,
        "batch_size": 128,
        "callbacks": {
            "monitor": {"metric": "val_accuracy", "max_or_min": "max"},
            "early_stop": {"patience": 5},
            "lr_decay": {"patience": 3},
        },
        "eps_start": 0.9,
        "eps_end": 0.05,
        "eps_decay": 200,
        "target_update": 10,
        "replay_experience_size": 10000,
        "num_episodes": 10000,
    },
    "strategies": {
        "episodic_memory_manage": "train",
        "episodic_question_answer": "latest",
        "semantic_memory_manage": "weakest",
        "semantic_question_answer": "strongest",
        "episodic_to_semantic": "find_common",
        "episodic_semantic_question_answer": "episodic_first",
        "pretrain_semantic": False,
        "capacity": {"episodic": 1, "semantic": 0},
        "policy_params": {"function_type": "mlp", "embedding_dim": 4},
    },
    "generator_params": {
        "max_history": 1024,
        "semantic_knowledge_path": "./data/semantic-knowledge.json",
        "names_path": "./data/top-human-names",
        "weighting_mode": "highest",
        "commonsense_prob": 0,
        "time_start_at": 0,
        "limits": {"heads": 2, "tails": 1, "names": 1, "allow_spaces": False},
        "disjoint_entities": True,
    },
}

seed_everything(config["seed"])

env = MemoryEnv(**config["generator_params"])

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayExperience(object):
    def __init__(self, capacity):
        self.experience = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.experience.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.experience, batch_size)

    def __len__(self):
        return len(self.experience)


env.reset()

policy_net = create_policy_net(
    capacity=config["strategies"]["capacity"],
    policy_type="episodic_memory_manage",
    generator_params=config["generator_params"],
    **config["strategies"]["policy_params"],
)
policy_net.set_device(config["training_params"]["device"])

target_net = create_policy_net(
    capacity=config["strategies"]["capacity"],
    policy_type="episodic_memory_manage",
    generator_params=config["generator_params"],
    **config["strategies"]["policy_params"],
)

target_net.set_device(config["training_params"]["device"])

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(
    policy_net.parameters(), lr=config["training_params"]["learning_rate"]
)
experience = ReplayExperience(config["training_params"]["replay_experience_size"])

steps_done = 0
n_actions = policy_net.out_features


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = config["training_params"]["eps_end"] + (
        config["training_params"]["eps_start"] - config["training_params"]["eps_end"]
    ) * math.exp(-1.0 * steps_done / config["training_params"]["eps_decay"])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]],
            device=config["training_params"]["device"],
            dtype=torch.long,
        )


def optimize_model():
    if len(experience) < config["training_params"]["batch_size"]:
        return
    transitions = experience.sample(config["training_params"]["batch_size"])
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=config["training_params"]["device"],
        dtype=torch.bool,
    )

    if len([s for s in batch.next_state if s is not None]) == 0:
        return
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(
        config["training_params"]["batch_size"],
        device=config["training_params"]["device"],
    )
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    # Compute the expected Q values
    expected_state_action_values = (
        next_state_values * config["training_params"]["gamma"]
    ) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    print(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in range(config["training_params"]["num_episodes"]):
    # Initialize the environment and state
    env.reset()
    M_e = EpisodicMemory(config["strategies"]["capacity"]["episodic"])
    M_s = None

    while not M_e.is_kinda_full:
        partial_state, reward, done, info = env.step("foo")
        M_e.add(partial_state["observation"])

    state = policy_net.make_state(
        **{"M_e": M_e, "M_s": M_s, "question": partial_state["question"]}
    )
    rewards = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        import pdb

        pdb.set_trace()
        mem = M_e.entries[int(action.item())]
        assert M_e.is_kinda_full
        M_e.forget(mem)

        pred = M_e.answer_latest(partial_state["question"])

        next_partial_state, reward, done, info = env.step(pred)
        M_e.add(M_e.ob2epi(next_partial_state["observation"]))
        reward = torch.tensor([reward], device=config["training_params"]["device"])

        # Observe new state
        if done:
            next_state = None
        else:
            next_state = policy_net.make_state(
                **{
                    "M_e": M_e,
                    "M_s": M_s,
                    "question": next_partial_state["question"],
                }
            )

        # Store the transition in memory
        experience.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % config["training_params"]["target_update"] == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Complete")
env.close()
