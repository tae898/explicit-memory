import logging
import os

import gym
import numpy as np
from gym import spaces

from memory.constants import CORRECT, WRONG

from ..memory import EpisodicMemory, Memory, SemanticMemory
from .generator import OQAGenerator
from .space import MemorySpace

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class EpisodicMemoryManageEnv(gym.Env):
    """Custom episodic_memory_manage environment compatiable with the gym interface."""

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
        question_answer: str = "latest",
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": True,
        },
        **kwargs,
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
        self.step_counter = 0
        self.max_history = max_history

    def reset(self):
        self.oqag.reset()
        self.M_e.forget_all()
        self.step_counter = 0

        for _ in range(self.M_e.capacity + 1):
            ob, _ = self.oqag.generate_with_history(generate_qa=False)
            mem_epi = self.M_e.ob2epi(ob)
            self.M_e.add(mem_epi)

        state_numeric = self.observation_space.episodic_memory_system_to_numbers(
            self.M_e, self.M_e.capacity + 1
        )

        return state_numeric

    def step(self, action):

        assert self.M_e.is_kinda_full

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

        qa_epi = self.oqag.generate_question_answer()

        if self.question_answer == "latest":
            reward, pred, correct_answer = self.M_e.answer_latest(qa_epi)
        elif self.question_answer == "random":
            reward, pred, correct_answer = self.M_e.answer_random(qa_epi)
        elif self.question_answer == "RL_trained":
            raise NotImplementedError
        else:
            raise ValueError

        ob, _ = self.oqag.generate_with_history(generate_qa=False)
        mem_epi = self.M_e.ob2epi(ob)
        self.M_e.add(mem_epi)

        state_numeric = self.observation_space.episodic_memory_system_to_numbers(
            self.M_e, self.M_e.capacity + 1
        )

        self.step_counter += 1

        if self.step_counter == self.max_history:
            # print(self.step_counter)
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


class EpisodicQuestionAnswer(gym.Env):
    """Custom episodic_question_answer environment compatiable with gym interface."""

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        capacity: dict,
        max_history: int = 1024,
        semantic_knowledge_path: str = "./data/semantic-knowledge.json",
        names_path: str = "./data/top-human-names",
        weighting_mode: str = "highest",
        commonsense_prob: float = 0.5,
        memory_manage: str = "oldest",
        question_answer: str = "RL_train",
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": True,
        },
        **kwargs,
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
        assert memory_manage in ["random", "oldest", "RL_trained"]
        assert question_answer in ["random", "latest", "RL_train", "RL_trained"]

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
        self.n_actions = self.capacity["episodic"]
        self.action_space = spaces.Discrete(self.n_actions)
        self.M_e = EpisodicMemory(self.capacity["episodic"])
        space_type = "episodic_question_answer"

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
        self.step_counter = 0
        self.max_history = max_history

    def reset(self):
        self.oqag.reset()
        self.M_e.forget_all()
        self.step_counter = 0

        for _ in range(self.M_e.capacity):
            ob, _ = self.oqag.generate_with_history(generate_qa=False)
            mem_epi = self.M_e.ob2epi(ob)
            self.M_e.add(mem_epi)
        self.qa_epi = self.oqag.generate_question_answer()

        state_numeric_1 = self.observation_space.episodic_memory_system_to_numbers(
            self.M_e, self.M_e.capacity
        )
        state_numeric_2 = self.observation_space.episodic_question_answer_to_numbers(
            self.qa_epi
        )

        state_numeric = np.concatenate([state_numeric_1, state_numeric_2])

        return state_numeric

    def step(self, action):

        assert self.M_e.is_full

        if self.question_answer == "latest":
            reward, pred, correct_answer = self.M_e.answer_latest(self.qa_epi)
        elif self.question_answer == "random":
            reward, pred, correct_answer = self.M_e.answer_random(self.qa_epi)
        elif self.question_answer == "RL_train":
            correct_answer = self.M_e.remove_name(self.qa_epi[2])
            pred = self.M_e.remove_name(self.M_e.entries[action][2])

            if pred == correct_answer:
                reward = CORRECT
            else:
                reward = WRONG
        elif self.question_answer == "RL_trained":
            raise NotImplementedError
        else:
            raise ValueError

        ob, self.qa_epi = self.oqag.generate_with_history(generate_qa=True)
        mem_epi = self.M_e.ob2epi(ob)
        self.M_e.add(mem_epi)

        assert self.M_e.is_kinda_full

        if self.memory_manage == "oldest":
            self.M_e.forget_oldest()
        elif self.memory_manage == "random":
            self.M_e.forget_random()
        elif self.memory_manage == "RL_trained":
            raise NotImplementedError
        else:
            raise ValueError

        state_numeric_1 = self.observation_space.episodic_memory_system_to_numbers(
            self.M_e, self.M_e.capacity
        )
        state_numeric_2 = self.observation_space.episodic_question_answer_to_numbers(
            self.qa_epi
        )

        state_numeric = np.concatenate([state_numeric_1, state_numeric_2])

        self.step_counter += 1

        if self.step_counter == self.max_history:
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


class SemanticMemoryManageEnv(gym.Env):
    """Custom semantic_memory_manage environment compatiable with the gym interface."""

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
        question_answer: str = "strongest",
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": True,
        },
        **kwargs,
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
        memory_manage: either "random", "weakset", "RL_train", or "RL_trained". Note that at
            this point `memory_manage` and `question_answer` can't be "RL" at the same
            time.
        question_answer: either "random", "strongest", "RL_trained". Note that at
            this point `memory_manage` and `question_answer` can't be "RL" at the same
            time.
        limits: Limit the heads, tails per head, and the number of names. For example,
            this can be {"heads": 10, "tails": 1, "names" 10, "allow_spaces": True}

        """
        super().__init__()
        assert memory_manage in ["random", "weakest", "RL_train", "RL_trained"]
        assert question_answer in ["random", "strongest", "RL_trained"]

        self.memory_manage = memory_manage
        self.question_answer = question_answer
        assert capacity["episodic"] == 0
        self.capacity = capacity
        self.oqag = OQAGenerator(
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            limits=limits,
        )
        self.n_actions = self.capacity["semantic"] + 1
        self.action_space = spaces.Discrete(self.n_actions)
        self.M_s = SemanticMemory(self.capacity["semantic"])
        space_type = "semantic_memory_manage"

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
        self.step_counter = 0
        self.max_history = max_history

    def reset(self):
        self.oqag.reset()
        self.M_s.forget_all()
        self.step_counter = 0

        while not self.M_s.is_kinda_full:
            ob, _ = self.oqag.generate_with_history(generate_qa=False)
            mem_sem = self.M_s.ob2sem(ob)
            self.M_s.add(mem_sem)

        state_numeric = self.observation_space.semantic_memory_system_to_numbers(
            self.M_s, self.M_s.capacity + 1, pad=False
        )

        return state_numeric

    def step(self, action):

        assert self.M_s.is_kinda_full

        if self.memory_manage == "weakest":
            self.M_s.forget_weakest()
        elif self.memory_manage == "random":
            self.M_s.forget_random()
        elif self.memory_manage == "RL_train":
            mem = self.M_s.entries[action]
            self.M_s.forget(mem)
        elif self.memory_manage == "RL_trained":
            raise NotImplementedError
        else:
            raise ValueError

        qa_epi = self.oqag.generate_question_answer()
        qa_sem = self.M_s.eq2sq(qa_epi)

        if self.question_answer == "strongest":
            reward, pred, correct_answer = self.M_s.answer_strongest(qa_sem)
        elif self.question_answer == "random":
            reward, pred, correct_answer = self.M_s.answer_random(qa_epi)
        elif self.question_answer == "RL_trained":
            raise NotImplementedError
        else:
            raise ValueError

        while not self.M_s.is_kinda_full:
            ob, _ = self.oqag.generate_with_history(generate_qa=False)
            mem_sem = self.M_s.ob2sem(ob)
            self.M_s.add(mem_sem)

        state_numeric = self.observation_space.semantic_memory_system_to_numbers(
            self.M_s, self.M_s.capacity + 1, pad=False
        )

        self.step_counter += 1

        if self.step_counter == self.max_history:
            done = True
        else:
            done = False

        info = {}

        return state_numeric, reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        else:
            print(self.M_s.entries)

    def close(self):
        pass


class SemanticQuestionAnswerEnv(gym.Env):
    """Custom semantic_question_answer environment compatiable with gym interface."""

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        capacity: dict,
        max_history: int = 1024,
        semantic_knowledge_path: str = "./data/semantic-knowledge.json",
        names_path: str = "./data/top-human-names",
        weighting_mode: str = "highest",
        commonsense_prob: float = 0.5,
        memory_manage: str = "weakest",
        question_answer: str = "RL_train",
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": True,
        },
        **kwargs,
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
        memory_manage: either "random", "weakest", or "RL_trained". Note that at
            this point `memory_manage` and `question_answer` can't be "RL" at the same
            time.
        question_answer: either "random", "strongest", "RL_train", "RL_trained". Note that at
            this point `memory_manage` and `question_answer` can't be "RL" at the same
            time.
        limits: Limit the heads, tails per head, and the number of names. For example,
            this can be {"heads": 10, "tails": 1, "names" 10, "allow_spaces": True}

        """
        super().__init__()
        assert memory_manage in ["random", "weakest", "RL_trained"]
        assert question_answer in ["random", "strongest", "RL_train", "RL_trained"]

        self.memory_manage = memory_manage
        self.question_answer = question_answer
        assert capacity["episodic"] == 0
        self.capacity = capacity
        self.oqag = OQAGenerator(
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            limits=limits,
        )
        self.n_actions = self.capacity["semantic"]
        self.action_space = spaces.Discrete(self.n_actions)
        self.M_s = SemanticMemory(self.capacity["semantic"])
        space_type = "semantic_question_answer"

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
        self.step_counter = 0
        self.max_history = max_history

    def reset(self):
        self.oqag.reset()
        self.M_s.forget_all()
        self.step_counter = 0

        while not self.M_s.is_full:
            ob, _ = self.oqag.generate_with_history(generate_qa=False)
            mem_sem = self.M_s.ob2sem(ob)
            self.M_s.add(mem_sem)

        qa_epi = self.oqag.generate_question_answer()
        self.qa_sem = self.M_s.eq2sq(qa_epi)

        state_numeric_1 = self.observation_space.semantic_memory_system_to_numbers(
            self.M_s, self.M_s.capacity
        )
        state_numeric_2 = self.observation_space.semantic_question_answer_to_numbers(
            self.qa_sem
        )

        state_numeric = np.concatenate([state_numeric_1, state_numeric_2])

        return state_numeric

    def step(self, action):

        assert self.M_s.is_full

        if self.question_answer == "strongest":
            reward, pred, correct_answer = self.M_s.answer_strongest(self.qa_sem)
        elif self.question_answer == "random":
            reward, pred, correct_answer = self.M_s.answer_random(self.qa_sem)
        elif self.question_answer == "RL_train":
            correct_answer = self.qa_sem[2]
            pred = self.M_s.entries[action][2]

            if pred == correct_answer:
                reward = CORRECT
            else:
                reward = WRONG
        elif self.question_answer == "RL_trained":
            raise NotImplementedError
        else:
            raise ValueError

        while not self.M_s.is_kinda_full:
            ob, _ = self.oqag.generate_with_history(generate_qa=False)
            mem_sem = self.M_s.ob2sem(ob)
            self.M_s.add(mem_sem)

        qa_epi = self.oqag.generate_question_answer()
        self.qa_sem = self.M_s.eq2sq(qa_epi)

        if self.memory_manage == "weakest":
            self.M_s.forget_weakest()
        elif self.memory_manage == "random":
            self.M_s.forget_random()
        elif self.memory_manage == "RL_trained":
            raise NotImplementedError
        else:
            raise ValueError

        state_numeric_1 = self.observation_space.semantic_memory_system_to_numbers(
            self.M_s, self.M_s.capacity
        )
        state_numeric_2 = self.observation_space.semantic_question_answer_to_numbers(
            self.qa_sem
        )

        state_numeric = np.concatenate([state_numeric_1, state_numeric_2])

        self.step_counter += 1

        if self.step_counter == self.max_history:
            done = True
        else:
            done = False

        info = {}

        return state_numeric, reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        else:
            print(self.M_s.entries)

    def close(self):
        pass
