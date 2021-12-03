import argparse
import logging
import gym
import os
from collections import OrderedDict, deque, namedtuple
from typing import List, Tuple
from pprint import pformat

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from pytorch_lightning.utilities.seed import seed_everything
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from memory.utils import read_yaml

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args
    ----
    capacity: size of the buffer

    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args
        ----
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
    """Iterable Dataset containing the ExperienceBuffer which will be updated with
    new experiences during training.

    Args
    ----
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
        Args
        ----
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
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        Args
        ----
        net: DQN network
        epsilon: value to determine likelihood of taking a random action
        device: current device

        Returns
        -------
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

        Args
        ----
        net: DQN network
        epsilon: value to determine likelihood of taking a random action
        device: current device

        Returns
        -------
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
        self, env_params: dict, dqn_params: dict, training_params: dict
    ) -> None:
        """
        Args
        ----
        env_params:
            env: str
                e.g., "EpisodicMemoryManageEnv"
            capacity: dict
                e.g., {episodic: 256, semantic: 0}
            max_history: int
                e.g., 256
            semantic_knowledge_path: str
                e.g., "./data/semantic-knowledge.json"
            names_path: str
                e.g., "./data/top-human-names"
            weighting_mode: str
                e.g., "highest"
            commonsense_prob: float
                e.g., 0.5
            memory_manage: str,
                e.g., "oldest"
            question_answer: str
                e.g., "latest"
            limits: dict
                e.g., {"heads": 10, "tails": 1, "names": 5, "allow_spaces": false"

        dqn_params:
            dqn: str
                the deep q network model (e.g., MLP)

        training_params:
            gpus: int
                e.g., 1
            max_epochs: int
                e.g., 50
            val_check_interval: int
                e.g., 1
            batch_size: int
                size of the batches (e.g., 1)
            lr: float
                learning rate (e.g., 0.01)
            gamma: float
                discount factor (e.g., 0.99)
            sync_rate: int
                how many frames do we update the target network (e.g., 10)
            replay_size: int
                capacity of the replay buffer (e.g., 1000)
            warm_start_size: int
                how many samples do we use to fill our buffer at the start of training
                (e.g., 1000)
            eps_last_frame: int
                what frame should epsilon stop decaying (e.g., 1000)
            eps_start: float
                starting value of epsilon (e.g., 1.0)
            eps_end: float
                final value of epsilon (e.g., 0.01)
            episode_length: int
                max length of an episode (e.g., 200)
            warm_start_steps: int
                max episode reward in the environment (e.g., 1000)

        """
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.env_params["env"].lower() == "episodicmemorymanageenv":
            from memory.environment.gym import EpisodicMemoryManageEnv as Env

            num_rows = self.hparams.env_params["capacity"]["episodic"] + 1
            num_cols = 6
            num_actions = num_rows

        self.env = Env(**env_params)

        if self.hparams.dqn_params["dqn"].lower() == "mlp":
            from memory.model import MLP as DQN

        self.net = DQN(num_rows, num_cols, num_actions)
        self.target_net = DQN(num_rows, num_cols, num_actions)

        self.buffer = ReplayBuffer(self.hparams.training_params["replay_size"])
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.training_params["warm_start_steps"])

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up
        the replay buffer with experiences.

        Args
        ----
        steps: number of random steps to populate the buffer with

        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action
        as an output.

        Args
        ----
        x: environment state

        Returns
        -------
        q values

        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args
        ----
        batch: current mini batch of replay data

        Returns
        -------
        mse loss

        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = (
            self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = (
            next_state_values * self.hparams.training_params["gamma"] + rewards
        )

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved.

        Args
        ----
        batch: current mini batch of replay data
        batch_idx: batch number

        Returns
        -------
        Training loss and log metrics

        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.training_params["eps_end"],
            self.hparams.training_params["eps_start"]
            - self.global_step
            + 1 / self.hparams.training_params["eps_last_frame"],
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
        if self.global_step % self.hparams.training_params["sync_rate"] == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward)
            .to(device)
            .detach()
            .cpu()
            .numpy()
            .tolist(),
            "reward": torch.tensor(reward).to(device).detach().cpu().numpy().tolist(),
            "train_loss": loss.detach().cpu().numpy().tolist(),
        }
        status = {
            "steps": torch.tensor(self.global_step)
            .to(device)
            .detach()
            .cpu()
            .numpy()
            .tolist(),
            "total_reward": torch.tensor(self.total_reward)
            .to(device)
            .detach()
            .cpu()
            .numpy()
            .tolist(),
        }

        self.log("train_loss", log["train_loss"])
        self.log("train_total_reward", log["total_reward"])
        self.log("train_reward", log["reward"])
        self.log("train_steps", status["steps"])

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx) -> OrderedDict:
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx) -> OrderedDict:
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch: Tuple[Tensor, Tensor], batch_idx, prefix):
        """Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved.

        Args
        ----
        batch: current mini batch of replay data
        batch_idx: batch number

        Returns
        -------
        Training loss and log metrics

        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.training_params["eps_end"],
            self.hparams.training_params["eps_start"]
            - self.global_step
            + 1 / self.hparams.training_params["eps_last_frame"],
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
        if self.global_step % self.hparams.training_params["sync_rate"] == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward)
            .to(device)
            .detach()
            .cpu()
            .numpy()
            .tolist(),
            "reward": torch.tensor(reward).to(device).detach().cpu().numpy().tolist(),
            "train_loss": loss.detach().cpu().numpy().tolist(),
        }
        status = {
            "steps": torch.tensor(self.global_step)
            .to(device)
            .detach()
            .cpu()
            .numpy()
            .tolist(),
            "total_reward": torch.tensor(self.total_reward)
            .to(device)
            .detach()
            .cpu()
            .numpy()
            .tolist(),
        }

        self.log(f"{prefix}_loss", log["train_loss"])
        self.log(f"{prefix}_total_reward", log["total_reward"])
        self.log(f"{prefix}_reward", log["reward"])
        self.log(f"{prefix}_steps", status["steps"])

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.training_params["episode_length"])
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.training_params["batch_size"],
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.training_params["lr"])
        return [optimizer]


def main(seed: int, env_params: dict, dqn_params: dict, training_params: dict) -> None:
    """Instantiate a LightningModule and Trainer, and start training.

    Args
    ----
    seed: global seed
    env_params: environment parameters
    dqn_parms: DQN parameters
    training_params: training parameters

    """
    seed_everything(seed)

    model = DQNLightning(
        env_params=env_params, dqn_params=dqn_params, training_params=training_params
    )

    trainer_params = {
        "gpus": training_params["gpus"],
        "max_epochs": training_params["max_epochs"],
        "val_check_interval": training_params["val_check_interval"],
        "log_every_n_steps": training_params["log_every_n_steps"],
    }

    trainer = Trainer(**trainer_params)

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train rl with arguments.")
    parser.add_argument(
        "--config", type=str, default="./train_RL.yaml", help="path to the config file."
    )
    args = parser.parse_args()

    config = read_yaml(args.config)
    logging.info(f"\nArguments\n---------\n{pformat(config,indent=4, width=1)}\n")

    main(**config)
