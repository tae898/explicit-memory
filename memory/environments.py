import logging
import os
import random
import time
from pprint import pformat
from typing import List, Tuple

from gym import spaces
from gym.spaces import Space

from .constants import CORRECT, MAX_INT_32, WRONG
from .memory import EpisodicMemory, Memory, SemanticMemory
from .utils import read_json

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class OQAGenerator:
    """Generate observation, question, and answer."""

    def __init__(
        self,
        max_history: int = 1024,
        semantic_knowledge_path: str = "./data/semantic-knowledge.json",
        names_path: str = "./data/top-human-names",
        weighting_mode: str = "highest",
        commonsense_prob: float = 0.5,
    ) -> None:
        """Initialize the generator.

        Args
        ----
        max_history: maximum history of observations.
        semantic_knowledge_path: path to the semantic knowledge generated from
            `collect_data.py`
        names_path: The path to the top 20 human name list.
        weighting_mode: "highest" chooses the one with the highest weight, "weighted"
            chooses all of them by weight, and null chooses every single one of them
            without weighting.
        commonsense_prob: the probability of an observation being covered by a
            commonsense

        """
        logging.debug("Creating an Observation-Question-Answer generator object ...")
        self.max_history = max_history
        self.history = []

        (
            self.semantic_knowledge,
            self.unique_heads,
            self.unique_relations,
            self.unique_tails,
        ) = self.load_semantic_knowledge(semantic_knowledge_path)
        self.heads = list(self.semantic_knowledge.keys())
        self.relations = list(
            set(
                [
                    key_
                    for key, val in self.semantic_knowledge.items()
                    for key_, val_ in val.items()
                ]
            )
        )
        self.tails = list(
            set(
                (
                    [
                        val__["tail"]
                        for key, val in self.semantic_knowledge.items()
                        for key_, val_ in val.items()
                        for val__ in val_
                    ]
                )
            )
        )

        self.names = self.read_names(names_path)
        self.weighting_mode = weighting_mode
        self.commonsense_prob = commonsense_prob

        logging.info("An Observation-Question-Answer generator object is generated!")

    def __eq__(self, other):
        eps = 0.01

        if self.max_history != other.max_history:
            return False

        for sh, oh in zip(self.history, other.history):
            if sh[0] != oh[0]:
                return False
            if sh[1] != oh[1]:
                return False
            if sh[2] != oh[2]:
                return False
            if abs(sh[3] - oh[3]) > eps:
                return False

        if self.heads != other.heads:
            return False
        if self.relations != other.relations:
            return False
        if self.tails != other.tails:
            return False
        if self.names != other.names:
            return False
        if self.weighting_mode != other.weighting_mode:
            return False
        if self.commonsense_prob != other.commonsense_prob:
            return False

        return True

    def reset(self) -> None:
        """Reset the generator."""
        logging.debug(f"Reseting the history of length {len(self.history)}...")
        self.history.clear()
        logging.info("Reseting the history is done!")

    @property
    def is_full(self) -> bool:
        """Return true if full."""

        return len(self.history) == self.max_history

    @staticmethod
    def load_semantic_knowledge(path: str) -> Tuple[list, list, list, list]:
        """Load saved semantic knowledge.

        Args
        ----
        path: the path to the pretrained semantic memory.

        Returns
        -------
        semantic_knowledge
        unique_heads
        unique_relations
        unique_tails

        """
        logging.debug(f"loading the semantic knowledge from {path}...")
        semantic_knowledge = read_json(path)

        unique_heads = sorted(list(set(semantic_knowledge.keys())))
        unique_tails = sorted(
            list(
                set(
                    [
                        foo["tail"]
                        for key, val in semantic_knowledge.items()
                        for key_, val_ in val.items()
                        for foo in val_
                    ]
                )
            )
        )
        unique_relations = sorted(
            list(
                set(
                    [
                        key_
                        for key, val in semantic_knowledge.items()
                        for key_, val_ in val.items()
                    ]
                )
            )
        )
        logging.info(f"semantic knowledge successfully loaded from {path}!")

        return semantic_knowledge, unique_heads, unique_relations, unique_tails

    @staticmethod
    def read_names(path: str = "./data/top-human-names") -> list:
        """Read 20 most common names.

        Args
        ----
        path: The path to the top 20 human name list.

        Returns
        -------
        names: human names (e.g., James)

        """
        logging.debug(f"Reading {path} ...")
        with open(path, "r") as stream:
            names = stream.readlines()
        names = [line.strip() for line in names]
        logging.info(f"Reading {path} complete! There are {len(names)} names in total")

        return names

    def generate_observation(self):
        """

        Returns
        --------
        ob: observation in [head, relation, tail, timestamp]

        """
        logging.debug("Generating an observation ...")

        head = random.choice(self.heads)
        relation = random.choice(self.relations)

        if random.random() < self.commonsense_prob:
            logging.debug(f"Generating a common location for {head} ...")
            tails = self.semantic_knowledge[head][relation]

            if self.weighting_mode == "weighted":
                tail = random.choices(
                    [tail["tail"] for tail in tails],
                    weights=[tail["weight"] for tail in tails],
                    k=1,
                )[0]
            elif self.weighting_mode == "highest":
                tail = sorted(
                    self.semantic_knowledge[head][relation], key=lambda x: x["weight"]
                )[-1]["tail"]
            else:
                raise ValueError
        else:
            logging.debug(f"Generating a NON common location for {head} ...")
            while True:
                tail = random.choice(self.tails)
                if tail not in self.semantic_knowledge[head][relation]:
                    break

        name = random.choice(self.names)
        posessive = "'s"

        head = name + posessive + " " + head
        tail = name + posessive + " " + tail

        # unix timestamp in seconds (including decimal points)
        timestamp = time.time()
        ob = [head, relation, tail, timestamp]
        logging.info(f"A new observation generated: {ob}")

        return ob

    def generate_question_answer(self) -> List[str]:
        """Generate a question based on the observation history.

        Returns
        -------
        question_answer: [head, relation, tail]. `head` and `relation` together make
             a question and `tail` is the answer.

        """
        assert len(self.history) != 0
        logging.debug("Generating a question and answer based on the history ...")
        heads_covered = []
        questions = []
        for ob in self.history[::-1]:
            head = ob[0]
            if head not in heads_covered:
                questions.append(ob)
                heads_covered.append(head)

        # -1 removes the timestamp
        question_answer = random.choice(questions)[:-1]
        logging.info(f"Generated question and answer is {question_answer}")

        return question_answer

    def add_observation_to_history(self, ob):
        """Add observation to history.

        Args
        ----
        ob: observation `[head, relation, tail, timestamp]`

        """
        logging.debug(
            f"Adding observation {ob} to history ... This is needed to "
            "generate a question later."
        )
        self.history.append(ob)
        logging.info(f"observation {ob} is added to history!")

    def generate(self, generate_qa: bool = True) -> Tuple[list, List[str]]:
        """Generate an observation, question, and answer.

        Everything comes from a uniform random distribution.

        Args
        ----
        generate_qa: Whether to generate a question and answer along with
            observation or not.

        Returns
        --------
        ob: observation `[head, relation, tail, timestamp]`
        qa: [head, relation, tail]. `head` and `relation` together make
             a question and `tail` is the answer

        """
        if self.is_full:
            raise ValueError(f"History {len(self.history)} is full")

        ob = self.generate_observation()
        self.add_observation_to_history(ob)
        logging.info("The new observation is added to the history.")
        if generate_qa:
            qa = self.generate_question_answer()
        else:
            qa = None

        return ob, qa

    def symbol2number(self, symbol: str):
        """Convert a symbol to a number

        pad: 0, answer: -1
        relations: from 100 to 999
        names: from 1000 to 9999
        heads: from 10,000 to 99,999
        tails: from 100,000 to 999,999

        """
        SPECIALS = {"pad": 0, "answer": -1}
        RELATIONS = {rel: 100 + idx for idx, rel in enumerate(sorted(self.relations))}
        NAMES = {name: 1000 + idx for idx, name in enumerate(sorted(self.names))}
        HEADS = {head: 10000 + idx for idx, head in enumerate(sorted(self.heads))}
        TAILS = {tail: 100000 + idx for idx, tail in enumerate(sorted(self.tails))}

        s2n = {**SPECIALS, **RELATIONS, **NAMES, **HEADS, **TAILS}

        return s2n[symbol]


class MemorySpace(Space):
    def __init__(
        self,
        capacity: dict,
        space_type: str,
        max_history: int = 1024,
        semantic_knowledge_path: str = "./data/semantic-knowledge.json",
        names_path: str = "./data/top-human-names",
        weighting_mode: str = "highest",
        commonsense_prob: float = 0.5,
    ) -> None:
        """

        Args
        ----
        capacity: memory capacity
            e.g., {'episodic': 42, 'semantic: 0}
        space_type: one of the below
            1. episodic_memory_manage
            2. episodic_question_answer
            3. semantic_memory_manage
            4. semantic_question_answer
            5. episodic_to_semantic
        max_history: maximum history of observations.
        semantic_knowledge_path: path to the semantic knowledge generated from
            `collect_data.py`
        names_path: The path to the top 20 human name list.
        weighting_mode: "highest" chooses the one with the highest weight, "weighted"
            chooses all of them by weight, and null chooses every single one of them
            without weighting.
        commonsense_prob: the probability of an observation being covered by a
            commonsense

        """
        assert space_type in [
            "episodic_memory_manage",
            "episodic_question_answer",
            "semantic_memory_manage",
            "semantic_question_answer",
            "episodic_to_semantic",
        ]

        if space_type == "episodic_memory_manage":
            assert capacity["episodic"] > 0 and capacity["semantic"] == 0

        elif space_type == "episodic_question_answer":
            assert capacity["episodic"] > 0 and capacity["semantic"] == 0

        elif space_type == "semantic_memory_manage":
            assert capacity["episodic"] == 0 and capacity["semantic"] > 0

        elif space_type == "semantic_question_answer":
            assert capacity["episodic"] == 0 and capacity["semantic"] > 0

        elif space_type == "episodic_to_semantic":
            assert capacity["episodic"] > 0 and capacity["semantic"] == 0

        self.space_type = space_type
        self.capacity = capacity

        self.oqag = OQAGenerator(
            max_history,
            semantic_knowledge_path,
            names_path,
            weighting_mode,
            commonsense_prob,
        )
        self.M_e = EpisodicMemory(self.capacity["episodic"])
        self.M_s = SemanticMemory(self.capacity["semantic"])

        super().__init__()

    def sample(self):
        """Sample a state."""
        self.oqag.reset()
        self.M_e.forget_all()
        self.M_s.forget_all()

        for _ in range(random.randint(1, self.M_e.capacity + 1)):
            ob, _ = self.oqag.generate(generate_qa=False)
            mem_epi = self.M_e.ob2epi(ob)
            self.M_e.add(mem_epi)
        qa = self.oqag.generate_question_answer()
        qa_epi = qa

        for _ in range(random.randint(1, self.M_s.capacity + 1)):
            ob, _ = self.oqag.generate(generate_qa=False)
            mem_sem = self.M_s.ob2sem(ob)
            self.M_s.add(mem_sem)
        qa = self.oqag.generate_question_answer()
        qa_sem = self.M_s.eq2sq(qa)

        state = {
            "episodic_memory_system": None,
            "episodic_question_answer": None,
            "semantic_memory_system": None,
            "semantic_question_answer": None,
        }
        if self.space_type == "episodic_memory_manage":
            state["episodic_memory_system"] = self.M_e.entries.copy()
            return state

        elif self.space_type == "episodic_question_answer":
            state["episodic_memory_system"] = self.M_e.entries.copy()
            state["episodic_question_answer"] = qa_epi
            return state

        elif self.space_type == "semantic_memory_manage":
            state["semantic_memory_system"] = self.M_s.entries.copy()
            return state

        elif self.space_type == "semantic_question_answer":
            state["semantic_memory_system"] = self.M_s.entries.copy()
            state["semantic_question_answer"] = qa_sem
            return state

        elif self.space_type == "episodic_to_semantic":
            state["episodic_memory_system"] = self.M_e.entries.copy()
            return state

        else:
            raise ValueError

    def correct_memory_system(self, memory_type: str, entries: list) -> bool:
        """Check if a given list is a legit memory system."""
        if not isinstance(entries, list):
            return False

        for entry in entries:
            if len(entry) != 4:
                return False
            if Memory.remove_name(entry[0]) not in self.oqag.unique_heads:
                return False
            if entry[1] not in self.oqag.unique_relations:
                return False
            if Memory.remove_name(entry[2]) not in self.oqag.unique_tails:
                return False
            if memory_type == "episodic":
                if not (0 < entry[3] <= MAX_INT_32):
                    return False
                if not isinstance(entry[3], float):
                    return False
                if entry[0].split()[0].split("'s")[0] not in self.oqag.names:
                    return False
                if entry[2].split()[0].split("'s")[0] not in self.oqag.names:
                    return False
            else:
                if not (1 <= entry[3] <= MAX_INT_32):
                    return False
                if not isinstance(entry[3], int):
                    return False

        return True

    def correct_question_answer(self, memory_type: str, qa: list) -> bool:
        """Check if a given question-answer is legit."""
        if not isinstance(qa, list):
            return False

        if len(qa) != 3:
            return False

        if Memory.remove_name(qa[0]) not in self.oqag.unique_heads:
            return False
        if qa[1] not in self.oqag.unique_relations:
            return False
        if Memory.remove_name(qa[2]) not in self.oqag.unique_tails:
            return False
        if memory_type == "episodic":
            if qa[0].split()[0].split("'s")[0] not in self.oqag.names:
                return False
            if qa[2].split()[0].split("'s")[0] not in self.oqag.names:
                return False

        return True

    def contains(self, x):

        if not isinstance(x, dict):
            return False

        if self.space_type == "episodic_memory_manage":
            must_have = ["episodic_memory_system"]

        elif self.space_type == "episodic_question_answer":
            must_have = ["episodic_memory_system", "episodic_question_answer"]

        elif self.space_type == "semantic_memory_manage":
            must_have = ["semantic_memory_system"]

        elif self.space_type == "semantic_question_answer":
            must_have = ["semantic_memory_system", "semantic_question_answer"]

        elif self.space_type == "episodic_to_semantic":
            must_have = ["episodic_memory_system"]

        else:
            raise ValueError

        for key, val in x.items():
            if key in must_have:
                if val is None:
                    return False
                if not isinstance(val, list):
                    return False
                if key == "episodic_memory_system":
                    if not self.correct_memory_system("episodic", val):
                        return False
                if key == "episodic_question_answer":
                    if not self.correct_question_answer("episodic", val):
                        return False
                if key == "semantic_memory_system":
                    if not self.correct_memory_system("semantic", val):
                        return False
                if key == "semantic_question_answer":
                    if not self.correct_question_answer("semantic", val):
                        return False
            else:
                if val is not None:
                    return False

        return True

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

    def __eq__(self, other):
        if self.capacity != other.capacity:
            return False
        if self.space_type != other.space_type:
            return False
        if self.oqag.max_history != other.oqag.max_history:
            return False
        if self.oqag.heads != other.oqag.heads:
            return False
        if self.oqag.relations != other.oqag.relations:
            return False
        if self.oqag.tails != other.oqag.tails:
            return False
        if self.oqag.names != other.oqag.names:
            return False
        if self.oqag.weighting_mode != other.oqag.weighting_mode:
            return False
        if self.oqag.commonsense_prob != other.oqag.commonsense_prob:
            return False

        return True
