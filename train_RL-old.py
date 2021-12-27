"""Copied a lot from:
https://pytorch-lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html"""
import argparse
import logging
import os
from collections import OrderedDict, deque, namedtuple
from pprint import pformat
from typing import List, Tuple

import gym
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from memory.utils import read_yaml

GLOBAL_COUNT = 0

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
    """Replay Buffer for storing past experiences allowing the agent to learn from
    them."""

    def __init__(self, replay_size: int) -> None:
        """
        Args
        ----
        replay_size: size of the buffer

        """
        self.buffer = deque(maxlen=replay_size)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args
        ----
        experience: tuple (state, action, reward, done, new_state)

        """
        self.buffer.append(experience)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample from the reply buffer.

        Args
        ----
        batch_size: batch size

        Returns
        -------
        states, actions, rewards, dones, next_states: the size is batch_size

        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices)
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with
    new experiences during training."""

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        """
        Args
        ----
        buffer: replay buffer
        sample_size: number of experiences to sample at a time

        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(
        self, env: gym.Env, replay_buffer: ReplayBuffer, override_time: bool = False
    ) -> None:
        """
        Args
        ----
        env: training environment
        replay_buffer: replay buffer storing experiences

        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.override_time = override_time
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Reset the environment and update the state."""
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
        global GLOBAL_COUNT
        GLOBAL_COUNT += 1
        if self.override_time:
            new_state, reward, done, _ = self.env.step(
                action, override_time=GLOBAL_COUNT
            )
        else:
            new_state, reward, done, _ = self.env.step(action, override_time=None)
        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self, seed: int, env_params: dict, dqn_params: dict, training_params: dict
    ) -> None:
        """
        Args
        ----
        seed: seed
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
                e.g., "random", "oldest", "RL_train", "RL_trained"
            question_answer: str
                e.g., "random", "latest", "RL_trained"
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
                the number of iterations (steps) between syncing up the target network
                with the train network
            replay_size: int
                the replay buffer size (e.g., 1000)
            warm_start_size: int
                how many random steps through the environment to be carried out at the
                start of training to fill the buffer with a starting point
                (e.g., 1000)
            eps_last_epoch: int
                the final epoch in for the decrease of epsilon. At this epoch
                espilon = eps_end (e.g., 5)
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
        self.agent = Agent(
            self.env, self.buffer, self.hparams.env_params["override_time"]
        )
        self.train_total_reward = 0
        self.train_episode_reward = 0
        self.train_accuracy = 0
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

    # def forward(self, x: Tensor) -> Tensor:
    #     """Passes in a state x through the network and gets the q_values of each action
    #     as an output.

    #     Args
    #     ----
    #     x: environment state

    #     Returns
    #     -------
    #     q values

    #     """
    #     output = self.net(x)
    #     return output

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
            - (self.current_epoch / self.hparams.training_params["eps_last_epoch"]),
        )

        self.log(
            "epsilon",
            epsilon,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.train_episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
            loss = loss.unsqueeze(0)

        if done:
            self.train_total_reward = self.train_episode_reward
            self.train_episode_reward = 0
            self.train_accuracy = (
                self.train_total_reward / self.hparams.env_params["max_history"]
            )

        # Soft update of target network
        if self.global_step % self.hparams.training_params["sync_rate"] == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        train_total_reward = (
            torch.tensor(self.train_total_reward)
            .to(device)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        train_reward = torch.tensor(reward).to(device).detach().cpu().numpy().tolist()

        train_loss = loss.detach().cpu().numpy().tolist()

        train_accuracy = self.train_accuracy

        # train_steps = torch.tensor(self.global_step).detach().cpu().numpy().tolist()

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_total_reward",
            train_total_reward,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_accuracy",
            train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_reward",
            train_reward,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        # self.log(
        #     "train_steps",
        #     train_steps,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        return loss

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.training_params["episode_length"])
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.hparams.training_params["batch_size"]
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.training_params["lr"])
        return [optimizer]


class ValidationCallback(Callback):
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:

        if pl_module.hparams.env_params["env"].lower() == "episodicmemorymanageenv":
            from memory.environment.gym import EpisodicMemoryManageEnv as Env

        env = Env(**pl_module.hparams.env_params)
        state_numeric = env.reset()
        state_numeric = torch.from_numpy(state_numeric).to(pl_module.device)

        val_rewards = 0
        with torch.no_grad():
            pl_module.net.eval()
            for _ in range(env.oqag.max_history):
                action = torch.argmax(pl_module.net(state_numeric))
                state_numeric, reward, done, info = env.step(action)
                state_numeric = torch.from_numpy(state_numeric).to(pl_module.device)
                val_rewards += reward

        val_accuracy = val_rewards / env.oqag.max_history

        pl_module.log(
            "val_accuracy",
            val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        pl_module.net.train()


def main(
    seed: int,
    env_params: dict,
    dqn_params: dict,
    callback_params: dict,
    trainer_params: dict,
    training_params: dict,
) -> None:
    """Instantiate a LightningModule and Trainer, and start training.

    Args
    ----
    seed: global seed
    env_params: environment parameters
    callback_params: callback function parameters
    dqn_parms: DQN parameters
    trainer_params: trainer parameters
    training_params: training parameters

    """
    assert training_params["batch_size"] == 1, "ONLY TESTED WITH BATCH SIZE 1 SO FAR."
    seed_everything(seed)

    checkpoint_callback = ModelCheckpoint(**callback_params["model_checkpoint"])
    my_callback = ValidationCallback()

    model = DQNLightning(
        seed=seed,
        env_params=env_params,
        dqn_params=dqn_params,
        training_params=training_params,
    )

    trainer = Trainer(callbacks=[my_callback, checkpoint_callback], **trainer_params)

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
