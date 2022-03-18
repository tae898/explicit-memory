"""Import necessary stuff and start training."""
import logging

logger = logging.getLogger()
logger.disabled = True

import math
import os
import random
from collections import deque, namedtuple
from copy import deepcopy
from itertools import count

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from memory import EpisodicMemory
from memory.environment.generator import OQAGenerator
from memory.environment.gym import EpisodicMemoryManageEnv
from memory.model import create_policy_net


def seed_everything(seed: int):
    """Seed every randomness to seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net([state]).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


seed_everything(1)
num_memories = 7
embedding_dim = 8
factor = 4
step_max = 256
max_history = 128
num_episodes = 100
BATCH_SIZE = 128
GAMMA = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10
replay_capacity = step_max
n_actions = num_memories

generator_params = {
    "max_history": max_history,
    "semantic_knowledge_path": "./data/semantic-knowledge.json",
    "names_path": "./data/top-human-names",
    "weighting_mode": "highest",
    "commonsense_prob": 0.5,
    "time_start_at": 0,
    "limits": {
        "heads": 40,
        "tails": 1,
        "names": 1,
        "allow_spaces": False,
    },
    "disjoint_entities": True,
}

model_params = {
    "function_type": "mlp",
    "capacity": {"episodic": num_memories, "semantic": 0},
    "policy_type": "episodic_memory_manage",
    "embedding_dim": embedding_dim,
    "generator_params": generator_params,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
M_e = EpisodicMemory(num_memories)
oqag = OQAGenerator(**generator_params)

policy_net = create_policy_net(deepcopy(model_params))
policy_net.set_device(device)
target_net = create_policy_net(deepcopy(model_params))
target_net.set_device(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


optimizer = optim.Adam(policy_net.parameters())
buffer = ReplayBuffer(replay_capacity)

steps_done = 0


def optimize_model():
    if len(buffer) < BATCH_SIZE:
        return
    transitions = buffer.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    non_final_next_states = [s for s in batch.next_state if s is not None]
    state_batch = batch.state
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # import pdb; pdb.set_trace()
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # import pdb; pdb.set_trace()

    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in range(num_episodes):
    # Initialize the environment and state
    # env.reset()
    # last_screen = get_screen()
    # current_screen = get_screen()
    # state = current_screen - last_screen
    while not M_e.is_kinda_full:
        ob, _ = oqag.generate_with_history(generate_qa=False)
        M_e.add(EpisodicMemory.ob2epi(ob))

    state = deepcopy({"M_e": M_e, "M_s": None, "question": None})
    rewards_episode = 0
    for t in count():
        assert M_e.is_kinda_full
        # Select and perform an action
        action = select_action(state)
        # _, reward, done, _ = env.step(action.item())
        # mem_to_forget = M_e.entries[action]
        # M_e.forget(mem_to_forget)
        M_e.forget_random()

        ob, qa = oqag.generate_with_history(generate_qa=True)

        question = qa[:2]
        answer = qa[2]

        pred = M_e.answer_latest(question)

        if str(pred).lower() == answer.lower():
            reward = 1
        else:
            reward = 0

        M_e.add(EpisodicMemory.ob2epi(ob))
        next_state = deepcopy({"M_e": M_e, "M_s": None, "question": None})

        if t >= step_max:
            done = True
        else:
            done = False

        rewards_episode += reward

        reward = torch.tensor([reward], device=device)

        if not done:
            pass
        else:
            next_state = None

        # Store the transition in buffer
        buffer.push(state, action, next_state, reward)
        # print(f"length of buffer {len(buffer)}")

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            # print(f"done!")
            # plot_durations()
            break
        # print(f"step {t} rewards: {rewards_episode}")
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    print(f"\nepisode {i_episode} rewards_episode: {rewards_episode}\n")

print("Complete")
