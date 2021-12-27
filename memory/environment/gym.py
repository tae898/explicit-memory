import logging
import os

import gym
from gym import spaces

from ..memory import EpisodicMemory, Memory, SemanticMemory
from .generator import OQAGenerator
from .space import MemorySpace

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        memory_manage: str = "RL_train",
        question_answer: str = "hand_crafted",
        limits: dict = None,
        **kwargs
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
        assert memory_manage in ["random", "oldest", "RL_train", "RL_trained"]
        assert question_answer in ["random", "latest", "RL_trained"]

        self.memory_manage = memory_manage
        self.question_answer = question_answer
        assert capacity["semantic"] == 0
        self.capacity = capacity
        self.oqag = OQAGenerator(
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            limits=limits,
        )
        self.n_actions = self.capacity["episodic"] + 1
        self.action_space = spaces.Discrete(self.n_actions)
        self.M_e = EpisodicMemory(self.capacity["episodic"])
        space_type = "episodic_memory_manage"

        self.observation_space = MemorySpace(
            capacity=capacity,
            space_type=space_type,
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            limits=limits,
        )

    def reset(self):
        self.oqag.reset()
        self.M_e.forget_all()

        state_numeric = self.observation_space.episodic_memory_system_to_numbers(
            self.M_e, self.M_e.capacity + 1
        )

        return state_numeric

    def step(self, action, override_time=None):

        if self.M_e.is_kinda_full:
            if self.memory_manage == "oldest":
                self.M_e.forget_oldest()
            elif self.memory_manage == "random":
                self.M_e.forget_random()
            elif self.memory_manage == "RL_train":
                mem = self.M_e.entries[action]
                self.M_e.forget(mem)
            elif self.memory_manage == "RL_trained":
                raise NotImplementedError
            else:
                raise ValueError

        if len(self.oqag.history) == 0:
            # HACK
            reward, pred, correct_answer = 1, None, None
        else:
            qa = self.oqag.generate_question_answer()

            if self.question_answer == "latest":
                reward, pred, correct_answer = self.M_e.answer_latest(qa)
            elif self.question_answer == "random":
                reward, pred, correct_answer = self.M_e.answer_random(qa)
            elif self.question_answer == "RL_trained":
                raise NotImplementedError
            else:
                raise ValueError

        ob, _ = self.oqag.generate(generate_qa=True, override_time=override_time)
        mem_epi = self.M_e.ob2epi(ob)
        self.M_e.add(mem_epi)

        state_numeric = self.observation_space.episodic_memory_system_to_numbers(
            self.M_e, self.M_e.capacity + 1
        )

        if self.oqag.is_full:
            done = True
        else:
            done = False

        info = {}

        return state_numeric, reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        else:
            print(self.M_e.entries)

    def close(self):
        pass
