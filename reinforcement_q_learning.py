import math
import os
import random
from collections import deque, namedtuple
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


def seed_everything(seed: int):
    """Seed every randomness to seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor) -> None:
    """Create sinusoidal embeddings, as in "Attention is All You Need".

    Copied from https://github.com/huggingface/transformers/blob/
    455c6390938a5c737fa63e78396cedae41e4e87e/src/transformers/modeling_distilbert.py#L53

    This is done in place.

    Args
    ----
    n_pos: number of positions (e.g., number of tokens)
    dim: size of the positional embedding vector (e.g., normally 768 is used for LMs. )
    out: torch weight to overwrite to.

    """
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()  # I don't think this is necessary?


class MyEnv(gym.Env):

    metadata = {"render.modes": ["console"]}

    def __init__(self, num_actions: int, step_max: int, embedding_dim: int):
        super().__init__()
        self.step_max = step_max
        self.step_counter = 0
        self.population = [i for i in range(0, num_actions)]
        self.weights = [i + 1 for i in range(0, num_actions)]

        self.positional_embeddings = nn.Embedding(num_actions, embedding_dim)

        create_sinusoidal_embeddings(
            n_pos=num_actions,
            dim=embedding_dim,
            out=self.positional_embeddings.weight,
        )
        self.state = self.positional_embeddings.weight.flatten().numpy()
        # self.state = np.array(self.population, dtype=np.float32)

    def sample(self):
        return random.choices(population=self.population, weights=self.weights, k=1)[0]

    def reset(self):
        self.step_counter = 0
        return self.state

    def step(self, action):
        if int(self.sample()) == int(action):
            reward = 1

        else:
            reward = 0

        self.step_counter += 1

        if self.step_counter >= self.step_max:
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        else:
            pass

    def close(self):
        pass


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


class MyDQN(nn.Module):
    def __init__(self, num_features: int, factor: int, embedding_dim: int):
        super().__init__()
        in_features = num_features * embedding_dim
        mid_features = num_features * embedding_dim * factor
        out_features = num_features

        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(mid_features, mid_features)
        self.fc3 = nn.Linear(mid_features, out_features)

        # self.bn1 = nn.BatchNorm1d(num_features * factor)
        # self.bn2a = nn.BatchNorm1d(num_features * factor)
        # self.bn2b = nn.BatchNorm1d(num_features * factor)
        # self.bn2c = nn.BatchNorm1d(num_features * factor)
        # self.bn3 = nn.BatchNorm1d(num_features)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        print(x.cpu().detach().numpy().mean(axis=0).round(3))

        return x


seed_everything(1)
num_memories = 20
embedding_dim = 8
factor = 4
step_max = 1000
num_episodes = 100
BATCH_SIZE = 64
GAMMA = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 10
replay_capacity = num_episodes * step_max
n_actions = num_memories

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
env = MyEnv(num_actions=num_memories, step_max=step_max, embedding_dim=embedding_dim)
env.reset()

policy_net = MyDQN(
    num_features=num_memories, embedding_dim=embedding_dim, factor=factor
).to(device)
target_net = MyDQN(
    num_features=num_memories, embedding_dim=embedding_dim, factor=factor
).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(replay_capacity)


steps_done = 0


episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
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
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in range(num_episodes):
    # Initialize the environment and state
    # env.reset()
    # last_screen = get_screen()
    # current_screen = get_screen()
    # state = current_screen - last_screen
    state = env.reset()
    state = torch.tensor([state])
    rewards_episode = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        # _, reward, done, _ = env.step(action.item())
        next_state, reward, done, _ = env.step(action.item())
        rewards_episode += reward
        next_state = torch.tensor([next_state])

        reward = torch.tensor([reward], device=device)

        # Observe new state
        # last_screen = current_screen
        # current_screen = get_screen()
        if not done:
            # next_state = current_screen - last_screen
            pass
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # print(f"length of memory {len(memory)}")

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            # print(f"done!")
            episode_durations.append(t + 1)
            # plot_durations()
            break
        # print(f"step {t} rewards: {rewards_episode}")
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    print(f"episode {i_episode} rewards_episode: {rewards_episode}")

print("Complete")
env.render()
env.close()
