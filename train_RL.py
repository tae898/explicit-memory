import gym
from gym import spaces
from memory.constants import CORRECT, WRONG
from memory.environments import OQAGenerator, MemorySpace
from memory import Memory, EpisodicMemory, SemanticMemory


class EpisodicMemoryManageEnv(gym.Env):
    """Custom Memory environment compatiable with the gym interface."""

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        capacity: dict,
        max_history: int = 1024,
        semantic_knowledge_path: str = "./data/semantic-knowledge.json",
        names_path: str = "./data/top-human-names",
        weighting_mode: str = "highest",
        commonsense_prob: float = 0.5,
        memory_manage: str = "hand_crafted",
        question_answer: str = "hand_crafted",
    ) -> None:
        """

        Args
        ----
        capacity: memory capacity
            e.g., {'episodic': 42, 'semantic: 0}
        max_history: maximum history of observations.
        semantic_knowledge_path: path to the semantic knowledge generated from
            `collect_data.py`
        names_path: The path to the top 20 human name list.
        weighting_mode: "highest" chooses the one with the highest weight, "weighted"
            chooses all of them by weight, and null chooses every single one of them
            without weighting.
        commonsense_prob: the probability of an observation being covered by a
            commonsense
        memory_manage: either "hand_crafted", "RL_train", or "RL_trained". Note that at
            this point `memory_manage` and `question_answer` can't be "RL" at the same
            time.
        question_answer: either "hand_crafted", "RL_trained". Note that at
            this point `memory_manage` and `question_answer` can't be "RL" at the same
            time.

        """
        super().__init__()
        assert memory_manage in ["hand_crafted", "RL_train", "RL_trained"]
        assert question_answer in ["hand_crafted", "RL_train", "RL_trained"]
        assert not (memory_manage == "RL_train" and question_answer == "RL_train")

        self.memory_manage = memory_manage
        self.question_answer = question_answer
        assert capacity["semantic"] == 0
        self.capacity = capacity
        self.oqag = OQAGenerator(
            max_history,
            semantic_knowledge_path,
            names_path,
            weighting_mode,
            commonsense_prob,
        )
        self.n_actions = 139
        self.action_space = spaces.Discrete(self.n_actions)
        self.M_e = EpisodicMemory(self.capacity["episodic"])
        space_type = "episodic_question_answer"

        self.observation_space = MemorySpace(
            capacity,
            space_type,
            max_history,
            semantic_knowledge_path,
            names_path,
            weighting_mode,
            commonsense_prob,
        )
        # self.time_zero = 100000
        # self.time_delta = 100
        # self.counter = 0

    def reset(self):
        self.oqag.reset()
        self.M_e.forget_all()

        ob, qa = self.oqag.generate(generate_qa=True)
        mem_epi = self.M_e.ob2epi(ob)
        self.M_e.add(mem_epi)

        state_numeric_1 = self.observation_space.episodic_memory_system_to_numbers(
            self.M_e, self.M_e.capacity
        )
        state_numeric_2 = self.observation_space.episodic_question_answer_to_numbers(qa)

        next_state = np.concatenate([state_numeric_1, state_numeric_2])

        self.qa = qa

        return next_state

    def step(self, action):
        # import pdb; pdb.set_trace()

        # if self.M_e.is_kinda_full:
        #     if self.memory_manage == "hand_crafted":
        #         self.M_e.forget_oldest()
        #         # self.M_e.forget_random()
        #     # elif self.memory_manage == "RL_train":
        #     #     print(f"ACTION: {action}")
        #     #     mem = self.M_e.entries[action]
        #     #     self.M_e.forget(mem)
        #     # elif self.memory_manage == "RL_trained":
        #     #     pass
        #     else:
        #         raise ValueError

        # if self.question_answer == "hand_crafted":
        #     raise ValueError

        # elif self.question_answer == "RL_train":
        reward_, _, correct_answer = self.M_e.answer_latest(self.qa)

        pred = self.oqag.number2string(action + 10000)
        if pred == correct_answer:
            reward = CORRECT
        else:
            reward = WRONG

        # elif self.question_answer == "RL_trained":
        #     pass
        # else:
        #     raise ValueError

        ob, self.qa = self.oqag.generate(generate_qa=True)
        mem_epi = self.M_e.ob2epi(ob)
        self.M_e.add(mem_epi)

        # ob[-1] = np.float32(self.time_zero + (self.counter * self.time_delta))

        state_numeric_1 = self.observation_space.episodic_memory_system_to_numbers(
            self.M_e, self.M_e.capacity
        )
        state_numeric_2 = self.observation_space.episodic_question_answer_to_numbers(
            self.qa
        )

        next_state = np.concatenate([state_numeric_1, state_numeric_2])

        # mem_epi = self.M_e.ob2epi(ob)
        # self.M_e.add(mem_epi)

        if self.oqag.is_full:
            print(f"TAETAETAE DONE FULL")
            done = True
        else:
            done = False

        info = {}

        # next_state = self.observation_space.episodic_memory_system_to_numbers(
        #     self.M_e, self.me_max
        # )

        # import pdb; pdb.set_trace()

        # self.counter += 1
        return next_state, reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        else:
            print(self.M_e.entries)

    def close(self):
        pass


import os
from collections import OrderedDict, deque, namedtuple
from typing import List, Tuple

import gym
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())


class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, num_rows: int, num_cols: int, num_actions: int):
        """
        Args
        ----
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers

        """
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.bn_row = nn.BatchNorm1d(num_cols)
        self.bn_col = nn.BatchNorm1d(num_rows)
        self.dropout = nn.Dropout(0.5)

        self.LinearRow1 = nn.Linear(num_cols, num_cols)
        self.LinearRow2 = nn.Linear(num_cols, 1)

        self.LinearCol1 = nn.Linear(num_rows, num_rows)
        self.LinearCol2 = nn.Linear(num_rows, num_actions)

    def forward(self, x):
        # x = x.float()
        x = self.LinearRow1(x)
        x = self.relu(x)
        # x = self.bn_row(x)
        x = self.dropout(x)

        x = self.LinearRow1(x)
        x = self.relu(x)
        # x = self.bn_row(x)
        x = self.dropout(x)

        x = self.LinearRow2(x)
        x = self.relu(x)

        x = x.view(-1, self.num_rows)

        x = self.LinearCol1(x)
        x = self.relu(x)
        # x = self.bn_col(x)
        x = self.dropout(x)

        x = self.LinearCol1(x)
        x = self.relu(x)
        # x = self.bn_col(x)
        x = self.dropout(x)

        x = self.LinearCol2(x)

        return x


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices)
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        capacity: dict,
        memory_manage: str,
        question_answer: str,
        max_history: int,
        num_actions: int,
        batch_size: int = 1,
        lr: float = 1e-2,
        env: str = "CartPole-v0",
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
        warm_start_steps: int = 1000,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = EpisodicMemoryManageEnv(
            capacity=capacity,
            max_history=max_history,
            memory_manage=memory_manage,
            question_answer=question_answer,
        )
        num_rows = capacity["episodic"] + 1
        num_cols = 6
        # self.env = gym.make(self.hparams.env)
        # obs_size = self.env.observation_space.shape[0]
        # n_actions = self.env.action_space.n

        # self.net = DQN(obs_size, n_actions)
        # self.target_net = DQN(obs_size, n_actions)

        self.net = DQN(num_rows, num_cols, num_actions)
        self.target_net = DQN(num_rows, num_cols, num_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = (
            self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved.

        Args
        ----
        batch: current mini batch of replay data
        nb_batch: batch number

        Returns
        -------
        Training loss and log metrics

        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        )

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }
        print(log)

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


if __name__ == "__main__":

    model = DQNLightning(
        capacity={"episodic": 1024, "semantic": 0},
        memory_manage="hand_crafted",
        question_answer="RL_train",
        batch_size=4,
        num_actions=139,
        lr=1e-2,
        gamma=0.99,
        sync_rate=10,
        replay_size=1000,
        warm_start_size=1000,
        eps_last_frame=1000,
        eps_start=1.0,
        eps_end=0.01,
        episode_length=200,
        warm_start_steps=1000,
        max_history=1024,
    )

    trainer = Trainer(
        gpus=1,
        max_epochs=1000,
        val_check_interval=1,
    )

    trainer.fit(model)
