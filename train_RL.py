"""Deep Q Network."""
import argparse
import os
import shutil
from collections import OrderedDict
from datetime import datetime
from pprint import pformat
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pl_bolts.datamodules.experience_source import (Experience,
                                                    ExperienceSourceDataset)
from pl_bolts.losses.rl import dqn_loss
from pl_bolts.models.rl.common.agents import ValueAgent
from pl_bolts.models.rl.common.memory import MultiStepBuffer
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DataParallelPlugin, DDP2Plugin
from torch import Tensor, optim
from torch._C import Value
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from memory import memory
from memory.utils import read_yaml

if _GYM_AVAILABLE:
    from gym import Env
else:  # pragma: no cover
    warn_missing_pkg("gym")
    Env = object


class DQN(LightningModule):
    """Basic DQN Model.
    PyTorch Lightning implementation of `DQN <https://arxiv.org/abs/1312.5602>`_
    Paper authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves,
    Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller.
    Model implemented by:
        - `Donal Byrne <https://github.com/djbyrne>`
    Example:
        >>> from pl_bolts.models.rl.dqn_model import DQN
        ...
        >>> model = DQN("PongNoFrameskip-v4")
    Train::
        trainer = Trainer()
        trainer.fit(model)
    Note:
        This example is based on:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py
    Note:
        Currently only supports CPU and single GPU training with `accelerator=dp`
    """

    def __init__(
        self,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1000,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        replay_size: int = 100000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = 0,
        batches_per_epoch: int = 1000,
        batches_per_epoch_eval: int = 1000,
        n_steps: int = 1,
        env_params: dict = None,
        **kwargs,
    ):
        """
        Args
        ----
        env_name: gym environment tag
        eps_start: starting value of epsilon for the epsilon-greedy exploration
        eps_end: final value of epsilon for the epsilon-greedy exploration
        eps_last_frame: the final frame in for the decrease of epsilon. At this frame
            espilon = eps_end
        sync_rate: the number of iterations between syncing up the target network with
            the train network
        gamma: discount factor
        learning_rate: learning rate
        batch_size: size of minibatch pulled from the DataLoader
        replay_size: total capacity of the replay buffer
        warm_start_size: how many random steps through the environment to be carried out
            at the start of training to fill the buffer with a starting point
        avg_reward_len: how many episodes to take into account when calculating the avg
            reward
        min_episode_reward: the minimum score that can be achieved in an episode. Used
            for filling the avg buffer before training begins
        batches_per_epoch: number of batches per epoch
        batches_per_epoch_eval: number of batches per epoch for validation and test.
            This is basically number of episodes for eval.
        n_steps: size of n step look ahead
        env_params:
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
            memory_manage: either "random", "oldest", "RL_train", or "RL_trained". Note that at
                this point `memory_manage` and `question_answer` can't be "RL" at the same
                time.
            question_answer: either "random", "latest", "RL_trained". Note that at
                this point `memory_manage` and `question_answer` can't be "RL" at the same
                time.
            limits: Limit the heads, tails per head, and the number of names. For example,
                this can be {"heads": 10, "tails": 1, "names" 10, "allow_spaces": True}

        """
        super().__init__()

        # Environment
        self.exp = None
        self.make_environments(**env_params)

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
        )

        # Hyperparameters
        self.sync_rate = sync_rate
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.batches_per_epoch = batches_per_epoch
        self.batches_per_epoch_eval = batches_per_epoch_eval
        self.n_steps = n_steps

        # self.save_hyperparameters()

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        for _ in range(avg_reward_len):
            self.total_rewards.append(
                torch.tensor(min_episode_reward, device=self.device)
            )

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

        self.state = self.env.reset()

    def run_n_episodes(
        self, env, n_epsiodes: int = 1, epsilon: float = 1.0
    ) -> List[int]:
        """Carries out N episodes of the environment with the current agent.

        Args
        ----
        env: environment to use, either train environment or test environment
        n_epsiodes: number of episodes to run
        epsilon: epsilon value for DQN agent

        """
        total_rewards = []
        for _ in range(n_epsiodes):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.agent.epsilon = epsilon
                action = self.agent(episode_state, self.device)
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience."""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in range(warm_start):
                self.agent.epsilon = 1.0
                action = self.agent(self.state, self.device)
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(
                    state=self.state,
                    action=action[0],
                    reward=reward,
                    done=done,
                    new_state=next_state,
                )
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()

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

    def train_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the
        DataLoader.

        Returns
        -------
        yields a Experience tuple containing the state, action, reward, done and
        next_state.

        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            action = self.agent(self.state, self.device)

            next_state, r, is_done, _ = self.env.step(action[0])

            episode_reward += r
            episode_steps += 1

            exp = Experience(
                state=self.state,
                action=action[0],
                reward=r,
                done=is_done,
                new_state=next_state,
            )

            self.agent.update_epsilon(self.global_step)
            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(
                    np.mean(self.total_rewards[-self.avg_reward_len :])
                )
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(
                self.batch_size
            )

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[
                    idx
                ]

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def eval_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """This is a dummy function for evaluation. This is needed to simulate batch

        Returns
        -------
        yields a Experience tuple containing the state, action, reward, done and
        next_state.

        """
        dummy_steps = 0
        while True:
            dummy_steps += 1

            for _ in range(self.batch_size):
                yield [], [], [], [], []

            # Simulates epochs
            if dummy_steps % self.batches_per_epoch_eval == 0:
                break

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay
        buffer. Then calculates loss based on the minibatch recieved.

        Args
        ----
        batch: current mini batch of replay data
        _: batch number, not used

        Returns
        -------
        Training loss and log metrics

        """
        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net, self.gamma)

        if self._use_dp_or_ddp2(self.trainer):
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "train_loss": loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
            }
        )

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Validate the agent for 1 episodes."""
        val_reward = self.run_n_episodes(self.val_env, 1, 0)
        avg_reward = sum(val_reward) / len(val_reward)
        return {"val_reward": avg_reward}

    def validation_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the val results."""
        print(
            f"\nlogging validation results that ran {self.batches_per_epoch_eval} epoch\n"
        )
        rewards = [x["val_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_val_reward", avg_reward)
        return {"avg_val_reward": avg_reward}

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 1 episodes."""
        test_reward = self.run_n_episodes(self.test_env, 1, 0)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        print(f"\nlogging test results that ran {self.batches_per_epoch_eval} epoch\n")
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self, train_mode: bool = True) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.replay_size, self.n_steps)
        self.populate(self.warm_start_size)

        if train_mode:
            self.dataset = ExperienceSourceDataset(self.train_batch)
        else:
            self.dataset = ExperienceSourceDataset(self.eval_batch)

        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader(train_mode=True)

    def val_dataloader(self) -> DataLoader:
        """Get val loader."""
        return self._dataloader(train_mode=False)

    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self._dataloader(train_mode=False)

    def build_networks(self, dqn: str) -> None:
        """Initializes the DQN train and target networks."""
        if dqn.lower() == "mlp":
            from memory.model import MLP as NeuralNetwork
        else:
            raise NotImplementedError

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.net = NeuralNetwork(self.obs_shape, self.n_actions)
        self.target_net = NeuralNetwork(self.obs_shape, self.n_actions)

    def make_environments(
        self,
        env_name: str,
        capacity: dict,
        max_history: int = 1024,
        semantic_knowledge_path: str = "./data/semantic-knowledge.json",
        names_path: str = "./data/top-human-names",
        weighting_mode: str = "highest",
        commonsense_prob: float = 0.5,
        memory_manage: str = "RL_train",
        question_answer: str = "latest",
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": True,
        },
        dqn: str = "mlp",
    ) -> Env:
        """Initialise gym  environment.

        Args
        ----
        env_name: environment name or tag
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
        memory_manage: either "random", "oldest", "RL_train", or "RL_trained". Note that at
            this point `memory_manage` and `question_answer` can't be "RL" at the same
            time.
        question_answer: either "random", "latest", "RL_trained". Note that at
            this point `memory_manage` and `question_answer` can't be "RL" at the same
            time.
        limits: Limit the heads, tails per head, and the number of names. For example,
            this can be {"heads": 10, "tails": 1, "names" 10, "allow_spaces": True}
        dqn: DQN model. E.g., MLP

        Returns
        -------
        gym environment

        """
        if env_name.lower() == "episodicmemorymanageenv":
            from memory.environment.gym import EpisodicMemoryManageEnv as Env
        elif env_name.lower() == "episodicquestionanswerenv":
            from memory.environment.gym import EpisodicQuestionAnswer as Env
        elif env_name.lower() == "semanticmemorymanageenv":
            from memory.environment.gym import SemanticMemoryManageEnv as Env
        elif env_name.lower() == "semanticquestionanswerenv":
            from memory.environment.gym import SemanticQuestionAnswerEnv as Env
        else:
            raise NotImplementedError

        self.env = Env(
            capacity=capacity,
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            memory_manage=memory_manage,
            question_answer=question_answer,
            limits=limits,
        )
        self.val_env = Env(
            capacity=capacity,
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            memory_manage=memory_manage,
            question_answer=question_answer,
            limits=limits,
        )
        self.test_env = Env(
            capacity=capacity,
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            memory_manage=memory_manage,
            question_answer=question_answer,
            limits=limits,
        )
        self.build_networks(dqn)

    @staticmethod
    def _use_dp_or_ddp2(trainer: Trainer) -> bool:
        return isinstance(
            trainer.training_type_plugin, (DataParallelPlugin, DDP2Plugin)
        )


def main(
    seed: int,
    dqn_params: dict,
    env_params: dict,
    callback_params: dict,
    trainer_params: dict,
    save_dir: str,
    model_summary: str,
    current_time: str,
) -> None:
    """Instantiate a LightningModule and Trainer, and start training.

    Args
    ----
    seed: global seed
    dqn_parms: DQN parameters
    env_params: environment parameters
    callback_params: callback function parameters
    trainer_params: trainer parameters
    model_summary: model summary so that you remember what it is.

    """
    seed_everything(seed)

    checkpoint_callback = ModelCheckpoint(**callback_params["model_checkpoint"])

    model = DQN(
        **dqn_params,
        env_params=env_params,
    )

    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=model_summary,
        version=current_time,
    )
    trainer = Trainer(callbacks=[checkpoint_callback], logger=logger, **trainer_params)

    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train rl with arguments.")
    parser.add_argument(
        "--config", type=str, default="./train_RL.yaml", help="path to the config file."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="lightning_logs",
        help="log and ckpt save dir",
    )
    parser.add_argument(
        "--model_summary",
        type=str,
        help="model summary so that you remember what it is.",
    )
    args = parser.parse_args()

    current_time = datetime.now().strftime(r"%m%d_%H%M%S")

    config = read_yaml(args.config)
    config_copy_dir = os.path.join(args.save_dir, args.model_summary, current_time)
    config_copy_dst = os.path.join(config_copy_dir, args.config)
    os.makedirs(config_copy_dir, exist_ok=True)
    shutil.copy(
        args.config,
        config_copy_dst,
    )

    print(f"\nArguments\n---------\n{pformat(config,indent=4, width=1)}\n")

    main(
        **config,
        save_dir=args.save_dir,
        model_summary=args.model_summary,
        current_time=current_time,
    )
