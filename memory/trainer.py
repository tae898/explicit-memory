import os
import logging
from typing import Tuple, List
from glob import glob

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


from .agent import Agent
from .environment.gym import MemoryEnv
from .utils import write_json, seed_everything, read_json, read_yaml
from .model import eps

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        seed: int,
        training_params: dict,
        strategies: dict,
        generator_params: dict,
        save_dir: str,
    ) -> None:
        """Instantiate a Trainer class object.

        Args
        ----
        seed: seed number, e.g., 42
        training_params: training parameters
            algorithm: policy_gradients
            device: "cuda" or "cpu"
            precision: 32
            num_processes: 16
            gamma: 0.99
            learning_rate: 0.0001
            batch_size: 1
            callbacks:
        strategies:
            episodic_memory_manage:
                oldest, random, train, or trained
            episodic_question_answer:
                latest, random, train, or trained
            semantic_memory_manage:
                weakest, random, train, or trained
            semantic_question_answer:
                strongest, random, train, or trained
            episodic_to_semantic:
                find_common, random, train, or trained
            episodic_semantic_question_answer:
                episodic_first, random, train, or trained
            capacity: e.g., {"episodic": 128, "semantic": 128},
        generator_params: OQAGenerator parameters
        save_dir: str

        """
        self.seed = seed

        self.training_params = training_params
        self.strategies = strategies
        self.generator_params = generator_params
        self.save_dir = save_dir

        self.agent = Agent(**strategies, generator_params=generator_params)
        self.env = MemoryEnv(**generator_params)

        self.writer = SummaryWriter(log_dir=save_dir)

    def run_episode(self, train_mode: bool) -> Tuple[float, float]:
        """Run one epsiode.

        Currently batch size is always one.

        Args
        ----
        train_mode: True if train, False if eval

        Returns
        -------
        ep_reward: total episode rewards
        acc: accuracy for one episode

        """
        for key, policy_net in self.agent.policy_nets.items():
            del policy_net.saved_actions[:]
        del self.agent.rewards[:]

        self.agent.reset()
        state = self.env.reset()
        ep_reward = 0
        acc = 0
        num_step = 0

        if train_mode:
            self.agent.set_train_mode()
        else:
            self.agent.set_eval_mode()

        print("Agent interacting with the environment until it's done...")
        while True:
            if train_mode:
                action = self.agent(state, num_step, train_mode)
            else:
                with torch.no_grad():
                    action = self.agent(state, num_step, train_mode)

            state, reward, done, _ = self.env.step(action)
            if train_mode:
                self.agent.rewards.append({"num_step": num_step, "reward": reward})
            ep_reward += reward

            num_step += 1
            acc = round(ep_reward / num_step, 4)

            if done:
                break

        return ep_reward, acc

    def run_callbacks(
        self,
        metrics_all: list,
        num_episode: int,
        monitor: dict,
        early_stop: dict = None,
        lr_decay: dict = None,
    ) -> bool:
        """Run callback functions.

        At the moment they are only called when one episode ends.

        Args
        ----
        metrics_all: list,
        num_episode: int,
        monitor: dict,
        early_stop: dict = None,
        lr_decay: dict = None,

        Returns
        -------
        stop_go: Whether or not to stop training.

        """
        if len(self.agent.policy_nets) == 0:
            return True

        self.save_models(metrics_all=metrics_all, num_episode=num_episode, **monitor)
        stop_go = False
        if early_stop is not None:
            stop_go, best_index = self.early_stopping(
                **early_stop,
                **monitor,
                metrics_all=metrics_all,
                num_episode=num_episode,
            )
        if lr_decay is not None:
            self.lr_decaying(
                **lr_decay, **monitor, metrics_all=metrics_all, num_episode=num_episode
            )

        if stop_go:
            self.delete_models(best_index)
            self.agent.load_policy_nets(self.save_dir, load_best=True)

        return stop_go

    def save_models(
        self,
        metrics_all: list,
        num_episode: int,
        metric: str,
        max_or_min: str,
        only_improved: bool = True,
    ) -> None:
        """Save all policy nets.

        Args
        ----
        metrics_all: list,
        num_episode: int,
        metric: str,
        max_or_min: str,
        only_improved: bool = True,

        """
        assert len(metrics_all) == num_episode + 1
        best_index = self.find_best_index(metrics_all, num_episode, metric, max_or_min)

        postfix = metrics_all[-1][metric]
        postfix = f"{num_episode}-{metric}-{postfix}"

        if only_improved:
            if best_index == num_episode:
                self.agent.save_models(self.save_dir, postfix)
        else:
            self.agent.save_models(self.save_dir, postfix)

    def delete_models(self, best_index: int) -> None:
        """Delete useless models.

        Args
        ----
        best_index: The index to keep. The others will be deleted.

        """
        for path in glob(os.path.join(self.save_dir, "*", "*.pth")):
            index = int(os.path.basename(path).split("-")[1])
            if index == best_index:
                os.rename(path, path.split(".pth")[0] + "-best.pth")
            else:
                os.remove(path)

    def lr_decaying(
        self,
        metrics_all: list,
        num_episode: int,
        patience: int,
        metric: str,
        max_or_min: str,
    ) -> None:
        """Learning rate decay.

        Args
        ----
        metrics_all: list,
        num_episode: int,
        patience: int,
        metric: str,
        max_or_min: str,

        """
        assert len(metrics_all) == num_episode + 1

        best_index = self.find_best_index(metrics_all, num_episode, metric, max_or_min)

        if len(metrics_all) - best_index >= patience:
            print(
                f"{metric} hasn't improved for consecutive {patience} episodes. "
                f"Learning rate decay by *0.1 ..."
            )
            self.optimizer.param_groups[0]["lr"] *= 0.1

            return True, best_index
        else:
            return False, None

    def early_stopping(
        self,
        metrics_all: list,
        num_episode: int,
        patience: int,
        metric: str,
        max_or_min: str,
    ) -> bool:
        """Whether to stop early or not.

        Args
        ----
        metrics: metrics
        patience: stop if the monitored metric does not improve for <patience> episodes.
        max_or_min: should you maximize or minimize the metric

        Returns
        -------
        True or False
        best_index

        """
        best_index = self.find_best_index(metrics_all, num_episode, metric, max_or_min)

        if len(metrics_all) - best_index >= patience:
            print(f"{metric} hasn't improved for {patience} episodes. Early stopping!")

            return True, best_index
        else:
            return False, None

    @staticmethod
    def find_best_index(
        metrics_all: list, num_episode: int, metric: str, max_or_min: str
    ) -> int:
        """Find the best index (the episode) from the metrics."""
        assert len(metrics_all) == num_episode + 1
        if max_or_min.lower() == "max":
            best_elem = max(metrics_all, key=lambda x: x[metric])
        elif max_or_min.lower() == "min":
            best_elem = min(metrics_all, key=lambda x: x[metric])
        else:
            raise ValueError

        best_index = metrics_all.index(best_elem)

        return best_index

    def train(self):
        """Start training."""
        seed_everything(self.seed)
        algo = self.training_params.pop("algorithm").lower()
        trainer_params = {**self.training_params, "save_dir": self.save_dir}
        if algo == "policy_gradients":
            self.train_policy_gradients(**trainer_params)
        elif algo == "dqn":
            raise NotImplementedError
        else:
            raise ValueError

    def test(self, seed: int = None):
        """Start testing."""
        print("Test starts ...")
        if seed is not None:
            seed_everything(seed)

        self.agent.load_policy_nets(self.save_dir, load_best=True)
        device = self.training_params["device"]
        self.agent.set_device(device)

        ep_reward, acc = self.run_episode(train_mode=False)
        print(f"test episode rewards: {ep_reward}\ttest accuracy: {acc}\n")
        metric = {}
        metric["test_episode_rewards"] = ep_reward
        metric["test_accuracy"] = acc

        results_path = os.path.join(self.save_dir, "results.json")
        if os.path.isfile(results_path):
            metrics_all = read_json(results_path)
        else:
            metrics_all = []

        metrics_all.append(metric)
        self.writer.add_scalar(tag="test_accuracy", scalar_value=acc)
        self.writer.add_scalar(
            tag="test_episode_rewards",
            scalar_value=ep_reward,
        )

        write_json(metrics_all, os.path.join(self.save_dir, "results.json"))

    def train_policy_gradients(
        self,
        device: str,
        precision: int,
        num_processes: int,
        gamma: float,
        learning_rate: float,
        batch_size: int,
        callbacks: dict,
        save_dir: str,
    ) -> None:
        """Train with policy gradients algorithm.

        Args
        ----

        """
        # Agent might have multiple policy nets to optimize.
        if len(self.agent.policy_nets) > 0:
            self.optimizer = optim.Adam(
                [
                    {"params": val.parameters()}
                    for key, val in self.agent.policy_nets.items()
                ],
                lr=learning_rate,
            )
        self.agent.set_device(device)

        metrics_all = []
        num_episode = 0
        while True:
            metric = {}
            print(f"Episode: {num_episode}\ttraining starts ...")
            ep_reward, acc = self.run_episode(train_mode=True)
            print(f"train episode rewards: {ep_reward}\ttrain accuracy: {acc}")
            metric["train_episode_rewards"] = ep_reward
            metric["train_accuracy"] = acc
            self.writer.add_scalar(
                tag="train_accuracy", scalar_value=acc, global_step=num_episode
            )
            self.writer.add_scalar(
                tag="train_episode_rewards",
                scalar_value=ep_reward,
                global_step=num_episode,
            )

            for key, policy_net in self.agent.policy_nets.items():
                if len(policy_net.saved_actions) == 0:
                    continue
                num_steps_used = [lp["num_step"] for lp in policy_net.saved_actions]
                R = 0
                policy_loss = []
                value_loss = []
                returns = []

                for r in self.agent.rewards[::-1]:
                    num_step = r["num_step"]
                    reward = r["reward"]

                    if num_step in num_steps_used:
                        R = reward + gamma * R
                        returns.insert(0, R)

                returns = torch.tensor(returns, dtype=torch.float32)
                returns = (returns - returns.mean()) / (returns.std() + eps)

                assert len(policy_net.saved_actions) == len(returns)

                for lp, R in zip(policy_net.saved_actions, returns):
                    log_prob = lp["log_prob"]
                    value = lp["state_value"]
                    advantage = R - value.item()

                    policy_loss.append(-log_prob * advantage)
                    value_loss.append(
                        F.smooth_l1_loss(value.squeeze(0), torch.tensor([R]))
                    )

                assert len(returns) == len(policy_loss) == len(value_loss)

                policy_loss = torch.cat(
                    [loss.unsqueeze(0) for loss in policy_loss]
                ).sum()
                value_loss = torch.cat([loss.unsqueeze(0) for loss in value_loss]).sum()

                loss_sum = policy_loss + value_loss
                print(f"train loss {key}: {loss_sum}")

                self.writer.add_scalar(
                    tag=f"train_policy_loss_{key}",
                    scalar_value=policy_loss,
                    global_step=num_episode,
                )

                print("backprop ...\n")
                self.optimizer.zero_grad()
                value_loss.backward()
                self.optimizer.step()

            print(f"Episode: {num_episode}\tvalidation starts ...")
            ep_reward, acc = self.run_episode(train_mode=False)
            print(f"val episode rewards: {ep_reward}\tval accuracy: {acc}\n")
            metric["val_episode_rewards"] = ep_reward
            metric["val_accuracy"] = acc
            self.writer.add_scalar(
                tag="val_accuracy", scalar_value=acc, global_step=num_episode
            )
            self.writer.add_scalar(
                tag="val_episode_rewards",
                scalar_value=ep_reward,
                global_step=num_episode,
            )

            metrics_all.append(metric)

            stop_go = self.run_callbacks(
                **callbacks, metrics_all=metrics_all, num_episode=num_episode
            )

            num_episode += 1

            if stop_go:
                print("training done.")
                break

        write_json(metrics_all, os.path.join(save_dir, "results.json"))
