import logging
import os
import random
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from .memory import EpisodicMemory, SemanticMemory
from .model import create_policy_net

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Agent:

    """Agent with episodic and semantic memory systems.

    Since MemoryEnv is POMDP, this agent constructs the states itself using memories.

    """

    def __init__(
        self,
        episodic_memory_manage: str = "oldest",
        episodic_question_answer: str = "latest",
        semantic_memory_manage: str = "weakest",
        semantic_question_answer: str = "strongest",
        episodic_to_semantic: str = "generalize",
        episodic_semantic_question_answer: str = "episem",
        pretrain_semantic: bool = False,
        capacity: dict = {"episodic": None, "semantic": None},
        model_params: dict = None,
        generator_params: dict = None,
    ) -> None:
        """Instantiate with parameters.

        Args
        ----
        episodic_memory_manage: str = "oldest",
        episodic_question_answer: str = "latest",
        semantic_memory_manage: str = "weakest",
        semantic_question_answer: str = "strongest",
        episodic_to_semantic: str = "generalize",
        episodic_semantic_question_answer: str = "episem",
        pretrain_semantic: bool = False,
        capacity: dict = {"episodic": 128, "semantic": 128},
        model_params: dict = None,
            function_type: mlp
            embedding_dim: 1
        generator_params: dict = None,
            max_history: int = 1024,
            semantic_knowledge_path: str = "./data/semantic-knowledge.json",
            names_path: str = "./data/top-human-names",
            weighting_mode: str = "highest",
            commonsense_prob: float = 0.5,
            time_start_at: int = 0,
            limits: dict = {
                "heads": None,
                "tails": None,
                "names": None,
                "allow_spaces": False,
            },
            disjoint_entities: bool = True,

        """
        self.episodic_memory_manage = episodic_memory_manage
        self.episodic_question_answer = episodic_question_answer
        self.semantic_memory_manage = semantic_memory_manage
        self.semantic_question_answer = semantic_question_answer
        self.episodic_to_semantic = episodic_to_semantic
        self.episodic_semantic_question_answer = episodic_semantic_question_answer
        self.pretrain_semantic = pretrain_semantic
        self.capacity = capacity
        self.model_params = model_params
        self.generator_params = generator_params

        self.M_e = EpisodicMemory(self.capacity["episodic"])
        self.M_s = SemanticMemory(self.capacity["semantic"])

        if self.pretrain_semantic:
            free_space = self.M_s.pretrain_semantic(**generator_params)
            self.M_e.increase_capacity(free_space)
            assert (self.M_e.capacity + self.M_s.capacity) == (
                capacity["episodic"] + capacity["semantic"]
            )

        self.episodic_memory_manage_policy = self.create_episodic_memory_manage_policy()
        self.episodic_question_answer_policy = (
            self.create_episodic_question_answer_policy()
        )
        self.semantic_memory_manage_policy = self.create_semantic_memory_manage_policy()
        self.semantic_question_answer_policy = (
            self.create_semantic_question_answer_policy()
        )
        self.episodic_to_semantic_policy = self.create_episodic_to_semantic_policy()
        self.episodic_semantic_question_answer_policy = (
            self.create_episodic_semantic_question_answer_policy()
        )

    def set_device(self, device: str = "cpu") -> None:
        """Set device to either cpu or cuda."""
        for key, val in self.policy_nets.items():
            val.set_device(device)

    def create_episodic_memory_manage_policy(self):
        """Create episodic memory manage policy.

        "oldest", "random", "train", or "trained"

        """
        if self.episodic_memory_manage == "oldest":

            def func(M_e: EpisodicMemory, num_step: int, train_mode: bool):
                M_e.forget_oldest()

            return func

        elif self.episodic_memory_manage == "random":

            def func(M_e: EpisodicMemory, num_step: int, train_mode: bool):
                M_e.forget_random()

            return func

        elif self.episodic_memory_manage == "train":

            raise NotImplementedError

        elif self.episodic_memory_manage == "trained":

            raise NotImplementedError

        else:
            raise TypeError

    def create_episodic_question_answer_policy(self):
        """Create episodic question answer policy.

        "latest", "random", "train", or "trained"

        """
        if self.episodic_question_answer == "latest":

            def func(
                M_e: EpisodicMemory, question: list, num_step: int, train_mode: bool
            ):
                pred = M_e.answer_latest(question)

                return pred

            return func

        elif self.episodic_question_answer == "random":

            def func(
                M_e: EpisodicMemory, question: list, num_step: int, train_mode: bool
            ):
                pred = M_e.answer_random(question)

                return pred

            return func

        elif self.episodic_question_answer == "train":

            raise NotImplementedError

        elif self.episodic_question_answer == "trained":

            raise NotImplementedError

        else:
            raise TypeError

    def create_semantic_memory_manage_policy(self):
        """Create semantic memory manage policy.

        "weakest", "random", "train", or "trained"

        """
        if self.semantic_memory_manage == "weakest":

            def func(M_s: SemanticMemory, num_step: int, train_mode: bool):
                M_s.forget_weakest()

            return func

        elif self.semantic_memory_manage == "random":

            def func(M_s: SemanticMemory, num_step: int, train_mode: bool):
                M_s.forget_random()

            return func

        elif self.semantic_memory_manage == "train":

            raise NotImplementedError

        elif self.semantic_memory_manage == "trained":

            raise NotImplementedError

        else:
            raise TypeError

    def create_semantic_question_answer_policy(self):
        """Create semantic question answer policy.

        "strongest", "random", "train", or "trained"

        """
        if self.semantic_question_answer == "strongest":

            def func(
                M_s: SemanticMemory, question: list, num_step: int, train_mode: bool
            ):
                pred = M_s.answer_strongest(question)

                return pred

            return func

        elif self.semantic_question_answer == "random":

            def func(
                M_s: SemanticMemory, question: list, num_step: int, train_mode: bool
            ):
                pred = M_s.answer_random(question)

                return pred

            return func

        elif self.semantic_question_answer == "train":

            raise NotImplementedError

        elif self.semantic_question_answer == "trained":

            raise NotImplementedError

        else:
            raise TypeError

    def create_episodic_to_semantic_policy(self):
        """Create episodic to semantic policy.

        "generalize", "noop", "train", or "trained"

        """
        if self.episodic_to_semantic == "generalize":

            def func(M_e: EpisodicMemory, num_step: int, train_mode: bool):
                mem_epis, mem_sem = M_e.get_similar()

                return mem_epis, mem_sem

            return func

        elif self.episodic_to_semantic == "noop":

            def func(M_e: EpisodicMemory, num_step: int, train_mode: bool):
                mem_epis, mem_sem = None, None

                return mem_epis, mem_sem

            return func

        elif self.episodic_to_semantic == "train":

            raise NotImplementedError

        elif self.episodic_to_semantic == "trained":

            raise NotImplementedError

        else:
            raise TypeError

    def create_episodic_semantic_question_answer_policy(self):
        """Create episodic and semantic question answer policy.

        "episem", "random", "train", or "trained"

        """
        if self.episodic_semantic_question_answer == "episem":

            def func(
                M_e: EpisodicMemory,
                M_s: SemanticMemory,
                question: list,
                num_step: int,
                train_mode: bool,
            ):
                if M_e.is_answerable(question):
                    pred = M_e.answer_latest(question)
                else:
                    pred = M_s.answer_strongest(question)
                return pred

            return func

        elif self.episodic_semantic_question_answer == "random":

            def func(
                M_e: EpisodicMemory,
                M_s: SemanticMemory,
                question: list,
                num_step: int,
                train_mode: bool,
            ):
                pred0 = M_e.answer_random(question)
                pred1 = M_s.answer_random(question)

                pred = random.choice([pred0, pred1])

                return pred

            return func

        elif self.episodic_semantic_question_answer == "train":

            raise NotImplementedError

        elif self.episodic_semantic_question_answer == "trained":

            raise NotImplementedError

        else:
            raise TypeError

    def reset(self) -> None:
        """Reset the agent. Forget all of the memories in both memory systems."""
        self.M_e.forget_all()
        self.M_s.forget_all()

    def set_eval_mode(self) -> None:
        """Set all policy nets to eval mode."""
        for key, model in self.policy_nets.items():
            model.eval()

    def set_train_mode(self) -> None:
        """Set all policy nets to train mode."""
        for key, model in self.policy_nets.items():
            model.train()

    def save_models(self, save_dir: str, postfix: str = "") -> None:
        """Save models to disk.

        Args
        ----
        save_dir: save directory
        postfix: a string value to add at the end of a file name.

        """
        for key, model in self.policy_nets.items():
            model_dir = os.path.join(save_dir, key)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{key}-{postfix}.pth")
            torch.save(model, model_path)

    def delete_policy_nets(self) -> None:
        """Remove the loaded policy nets from memory."""
        for key, val in self.policy_nets.items():
            del val
        self.policy_nets = {}

    def load_policy_nets(self, load_dir: str, load_best: bool = True) -> None:
        """Load policy nets from disk.

        Args
        ----
        load_dir: load directory
        load_best: whether to load the best checkpoint.

        """
        self.delete_policy_nets()

        model_dirs = list(
            set(
                [
                    os.path.dirname(path)
                    for path in glob(os.path.join(load_dir, "*", "*.pth"))
                ]
            )
        )
        for model_dir in model_dirs:
            if load_best:
                model_path = glob(os.path.join(model_dir, "*best*.pth"))
                assert len(model_path) == 1
                model_path = model_path[0]
            model = torch.load(model_path)
            model.eval()
            policy_type = model_dir.split("/")[-1]

            self.policy_nets[policy_type] = model

    def __call__(self, partial_state: dict, num_step: int, train_mode: bool) -> str:
        """Takes in the partial state given by the environment and take an action.

        Args
        ----
        partial_state: {"observation": ob, "question": question}
        num_step: step number in episode
        train_mode: True or False

        Returns
        -------
        pred: prediction to the question.

        """
        ob = partial_state["observation"]
        question = partial_state["question"]

        if self.capacity["episodic"] > 0 and self.capacity["semantic"] == 0:
            mem_epi = self.M_e.ob2epi(ob)
            self.M_e.add(mem_epi)

            if self.M_e.is_kinda_full:
                self.episodic_memory_manage_policy(self.M_e, num_step, train_mode)
                assert self.M_e.is_full

            pred = self.episodic_question_answer_policy(
                self.M_e, question, num_step, train_mode
            )

        elif self.capacity["episodic"] == 0 and self.capacity["semantic"] > 0:
            if not self.M_s.is_frozen:
                mem_sem = self.M_s.ob2sem(ob)
                self.M_s.add(mem_sem)

                if self.M_s.is_kinda_full:
                    self.semantic_memory_manage_policy(self.M_s, num_step, train_mode)
                    assert self.M_s.is_full

            pred = self.semantic_question_answer_policy(
                self.M_s, question, num_step, train_mode
            )

        elif self.capacity["episodic"] > 0 and self.capacity["semantic"] > 0:
            mem_epi = self.M_e.ob2epi(ob)
            self.M_e.add(mem_epi)

            if self.M_s.is_frozen:
                if self.M_e.is_kinda_full:
                    self.episodic_memory_manage_policy(self.M_e, num_step, train_mode)
                    assert self.M_e.is_full
            else:
                if self.M_e.is_kinda_full:

                    mem_epis, mem_sem = self.episodic_to_semantic_policy(
                        self.M_e, num_step
                    )

                    if mem_epis is not None:
                        self.M_s.add(mem_sem)
                        if self.M_s.is_kinda_full:
                            self.semantic_memory_manage_policy(
                                self.M_s, num_step, train_mode
                            )
                            assert self.M_s.is_full

                        for mem_epi_ in mem_epis:
                            self.M_e.forget(mem_epi_)
                    else:
                        self.episodic_memory_manage_policy(
                            self.M_e, num_step, train_mode
                        )
                        assert self.M_e.is_full
            pred = self.episodic_semantic_question_answer_policy(
                self.M_e, self.M_s, question, num_step, train_mode
            )
        else:
            raise ValueError

        return pred
