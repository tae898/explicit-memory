import logging
import os
import random
import time
from pprint import pformat
from typing import List, Tuple
import numpy as np
from gym.spaces import Space


from .constants import MAX_INT_32, TIME_OFFSET
from .memory import Memory, EpisodicMemory, SemanticMemory
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
        pad: int = 0,
        answer: int = -1,
        relations_start_at: int = 10,
        names_start_at: int = 100,
        heads_start_at: int = 1000,
        tails_start_at: int = 10000,
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
        pad: 0
        answer: -1
        relations: from 10 to 99
        names: from 100 to 999
        heads: from 1,000 to 9,999
        tails: from 10,000 to 99,999
        """
        logging.debug("Creating an Observation-Question-Answer generator object ...")
        self.max_history = max_history
        self.history = []

        (
            self.semantic_knowledge,
            self.heads,
            self.relations,
            self.tails,
        ) = self.load_semantic_knowledge(semantic_knowledge_path)

        self.names = self.read_names(names_path)
        self.weighting_mode = weighting_mode
        self.commonsense_prob = commonsense_prob

        self.SPECIALS = {"<pad>": pad, "<answer>": answer}
        self.RELATIONS = {
            rel: relations_start_at + idx
            for idx, rel in enumerate(sorted(self.relations))
        }
        self.NAMES = {
            name: names_start_at + idx for idx, name in enumerate(sorted(self.names))
        }
        self.HEADS = {
            head: heads_start_at + idx for idx, head in enumerate(sorted(self.heads))
        }
        self.TAILS = {
            tail: tails_start_at + idx for idx, tail in enumerate(sorted(self.tails))
        }

        self.s2n = {
            **self.SPECIALS,
            **self.RELATIONS,
            **self.NAMES,
            **self.HEADS,
            **self.TAILS,
        }
        self.n2s = {val: key for key, val in self.s2n.items()}

        logging.info("An Observation-Question-Answer generator object is generated!")

    def __eq__(self, other) -> bool:
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
        if self.s2n != other.s2n:
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
        heads
        relations
        tails

        """
        logging.debug(f"loading the semantic knowledge from {path}...")
        semantic_knowledge = read_json(path)

        heads = sorted(list(set(semantic_knowledge.keys())))
        tails = sorted(
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
        relations = sorted(
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

        return semantic_knowledge, heads, relations, tails

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

    def generate_observation(self) -> list:
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
        timestamp = time.time() - TIME_OFFSET
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

    def add_observation_to_history(self, ob) -> None:
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

    def string2number(self, string) -> int:
        """Convert a given string to a number.

        Args
        ----
        string: e.g., "laptop"

        Returns
        -------
        number: 1027

        """
        return self.s2n[string]

    def number2string(self, number) -> str:
        """Convert a given number to a string.

        Args
        ----
        number: e.g., 1027

        Returns
        -------
        string: e.g., "laptop"

        """
        return self.n2s[number]


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
            6. episodic_semantic_question_answer
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
            "episodic_semantic_question_answer",
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
            assert capacity["episodic"] > 0 and capacity["semantic"] > 0

        elif space_type == "episodic_semantic_question_answer":
            assert capacity["episodic"] > 0 and capacity["semantic"] > 0

        else:
            raise ValueError

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

    def episodic_memory_system_to_numbers(
        self, M_e: EpisodicMemory, me_max: int
    ) -> np.ndarray:
        """Convert a given episodic memory system to numbers.

        This conversion is necessary so that we can use a computational model.

        Args
        ----
        M_e: Episodic memory object
        me_max: maximum number of episodic memories (rows)

        Returns
        -------
        state_numeric: numpy array where every string is replaced with a number

        """
        logging.debug("Converting the episodic memory system to a numpy array ...")
        state_string = M_e.entries.copy()
        NUM_COLUMNS = 6
        state_numeric = np.zeros((me_max, NUM_COLUMNS), dtype=np.float32)

        for idx, row in enumerate(state_string):
            head = row[0]
            name1, obj = M_e.split_name_entity(head)
            relation = row[1]
            tail = row[2]
            name2, location = M_e.split_name_entity(tail)
            timestamp = np.float32(row[3])

            assert name1 == name2

            state_numeric[idx][0] = self.oqag.string2number(name1)
            state_numeric[idx][1] = self.oqag.string2number(obj)
            state_numeric[idx][2] = self.oqag.string2number(relation)
            state_numeric[idx][3] = self.oqag.string2number(name2)
            state_numeric[idx][4] = self.oqag.string2number(location)
            state_numeric[idx][5] = timestamp

        logging.info("The episodic memory system has been converted to a numpy array!")

        return state_numeric

    def numbers_to_episodic_memories(self, state_numeric: np.ndarray) -> list:
        """Convert a given numpy array to episodic memories.

        This is for debugging purposes.

        Args
        ----
        state_numeric: the shape should be (size, NUM_COLUMNS), where NUM_COLUMNS is
            most likely to be 6

        Returns
        -------
        entries: episodic memories, where every row is [head, relation, tail, timestamp]

        """
        logging.debug("Converting the numpy array to episodic memories ...")
        NUM_COLUMNS = 6
        assert state_numeric.shape[1] == NUM_COLUMNS
        entries = []
        for row in state_numeric:
            name1 = self.oqag.number2string(int(row[0]))
            obj = self.oqag.number2string(int(row[1]))
            relation = self.oqag.number2string(int(row[2]))
            name2 = self.oqag.number2string(int(row[3]))
            location = self.oqag.number2string(int(row[4]))
            timestamp = np.float32(row[5])

            assert name1 == name2

            if name1 == "<pad>":
                assert (
                    obj
                    == relation
                    == name2
                    == location
                    == self.oqag.number2string(int(timestamp))
                    == "<pad>"
                )
                continue

            entries.append(
                [f"{name1}'s {obj}", relation, f"{name1}'s {location}", timestamp]
            )

        logging.info("The numpy array has been converted to episodic memories!")

        return entries

    def semantic_memory_system_to_numbers(
        self, M_s: SemanticMemory, ms_max: int, pad: bool = False
    ) -> np.ndarray:
        """Convert a given semantic memory system to numbers.

        This conversion is necessary so that we can use a computational model.

        Args
        ----
        M_s: Semantic memory object
        ms_max: maximum number of semantic memories (rows)
        pad: This makes the number of columns 6, not 4.

        Returns
        -------
        state_numeric: numpy array where every string is replaced with a number

        """
        logging.debug("Converting the semantic memory system to a numpy array ...")

        if pad:
            NUM_COLUMNS = 6
            state_string = M_s.entries.copy()
            state_numeric = np.zeros((ms_max, NUM_COLUMNS), dtype=np.float32)

            for idx, row in enumerate(state_string):
                head = row[0]
                relation = row[1]
                tail = row[2]
                num_general = row[3]

                state_numeric[idx][0] = self.oqag.string2number("<pad>")
                state_numeric[idx][1] = self.oqag.string2number(head)
                state_numeric[idx][2] = self.oqag.string2number(relation)
                state_numeric[idx][3] = self.oqag.string2number("<pad>")
                state_numeric[idx][4] = self.oqag.string2number(tail)
                state_numeric[idx][5] = num_general
        else:
            NUM_COLUMNS = 4
            state_string = M_s.entries.copy()
            state_numeric = np.zeros((ms_max, NUM_COLUMNS), dtype=np.float32)

            for idx, row in enumerate(state_string):
                head = row[0]
                relation = row[1]
                tail = row[2]
                num_general = row[3]

                state_numeric[idx][0] = self.oqag.string2number(head)
                state_numeric[idx][1] = self.oqag.string2number(relation)
                state_numeric[idx][2] = self.oqag.string2number(tail)
                state_numeric[idx][3] = num_general

        logging.info("The semantic memory system has been converted to a numpy array!")

        return state_numeric

    def numbers_to_semantic_memories(self, state_numeric: np.ndarray) -> list:
        """Convert a given numpy array to semantic memories.

        This is for debugging purposes.

        Args
        ----
        state_numeric: the shape should be (size, NUM_COLUMNS), where NUM_COLUMNS is
            most likely to be 4

        Returns
        -------
        entries: semantic memories, where every row is [head, relation, tail,
            num_generalized]

        """
        logging.debug("Converting the numpy array to semantic memories ...")
        entries = []
        if state_numeric.shape[1] == 6:
            for row in state_numeric:
                name1 = self.oqag.number2string(int(row[0]))
                obj = self.oqag.number2string(int(row[1]))
                relation = self.oqag.number2string(int(row[2]))
                name2 = self.oqag.number2string(int(row[3]))
                location = self.oqag.number2string(int(row[4]))
                num_general = row[5]

                assert name1 == name2 == "<pad>"

                if obj == "<pad>":
                    assert (
                        obj
                        == relation
                        == location
                        == self.oqag.number2string(int(num_general))
                        == "<pad>"
                    )
                    continue

                entries.append([obj, relation, location, num_general])
        elif state_numeric.shape[1] == 4:
            for row in state_numeric:
                obj = self.oqag.number2string(int(row[0]))
                relation = self.oqag.number2string(int(row[1]))
                location = self.oqag.number2string(int(row[2]))
                num_general = row[3]

                if obj == "<pad>":
                    assert (
                        obj
                        == relation
                        == location
                        == self.oqag.number2string(int(num_general))
                        == "<pad>"
                    )
                    continue

                entries.append([obj, relation, location, num_general])

        logging.info("The numpy array has been converted to semantic memories!")

        return entries

    def episodic_question_answer_to_numbers(self, qa_epi: list) -> np.ndarray:
        """Convert a given episodic qa to numbers.

        This conversion is necessary so that we can use a computational model.

        Args
        ----
        qa_epi: episodic question and answer, [head, relation, tail]

        Returns
        -------
        qa_num: [name1, obj, relation, name2, location, <answer>]

        """
        logging.debug("Converting a given episodic qa into a numpy array ...")
        NUM_COLUMNS = 6
        qa_num = np.zeros((1, NUM_COLUMNS), dtype=np.float32)
        head = qa_epi[0]
        name1, obj = EpisodicMemory.split_name_entity(head)
        relation = qa_epi[1]
        tail = qa_epi[2]
        name2, location = EpisodicMemory.split_name_entity(tail)

        assert name1 == name2

        qa_num[0][0] = self.oqag.string2number(name1)
        qa_num[0][1] = self.oqag.string2number(obj)
        qa_num[0][2] = self.oqag.string2number(relation)
        qa_num[0][3] = self.oqag.string2number(name2)
        qa_num[0][4] = self.oqag.string2number(location)
        qa_num[0][5] = self.oqag.string2number("<answer>")

        logging.debug("The numpy array has been converted to an episodic qa!")

        return qa_num

    def numbers_to_episodic_question_answer(self, qa_num: np.ndarray) -> list:
        """Convert a given numpy array to an episodic qa.

        Args
        ----
        qa_num: [name1, obj, relation, name2, location, <answer>]


        Returns
        -------
        qa_epi: episodic question and answer, [head, relation, tail]

        """
        logging.debug("Converting the numpy array to an episodic qa ...")

        assert qa_num.shape == (1, 6)

        name1 = self.oqag.number2string(int(qa_num[0][0]))
        obj = self.oqag.number2string(int(qa_num[0][1]))
        relation = self.oqag.number2string(int(qa_num[0][2]))
        name2 = self.oqag.number2string(int(qa_num[0][3]))
        location = self.oqag.number2string(int(qa_num[0][4]))
        answer = self.oqag.number2string(int(qa_num[0][5]))

        assert answer == "<answer>"
        assert name1 == name2

        qa_epi = [f"{name1}'s {obj}", relation, f"{name2}'s {location}"]

        logging.info("The numpy array has been converted to an episodic qa!")

        return qa_epi

    def semantic_question_answer_to_numbers(self, qa_sem: list) -> np.ndarray:
        """Convert a given semantic qa to numbers.

        This conversion is necessary so that we can use a computational model.

        Args
        ----
        qa_sem: semantic question and answer, [head, relation, tail]

        Returns
        -------
        qa_num: [obj, relation, location, <answer>]

        """
        logging.debug("Converting a given semantic qa into a numpy array ...")

        qa_num = np.zeros((1, 4), dtype=np.float32)
        head = qa_sem[0]
        relation = qa_sem[1]
        tail = qa_sem[2]

        qa_num[0][0] = self.oqag.string2number(head)
        qa_num[0][1] = self.oqag.string2number(relation)
        qa_num[0][2] = self.oqag.string2number(tail)
        qa_num[0][3] = self.oqag.string2number("<answer>")

        logging.debug("The given semantic qa has been converted to a numpy array!")

        return qa_num

    def numbers_to_semantic_question_answer(self, qa_num: np.ndarray) -> list:
        """Convert a given numpy array to an semantic qa.

        Args
        ----
        qa_num: [obj, relation, location, <answer>]


        Returns
        -------
        qa_sem: semantic question and answer, [head, relation, tail]

        """
        logging.debug("Converting the numpy array to an episodic qa ...")
        assert qa_num.shape == (1, 4)

        obj = self.oqag.number2string(int(qa_num[0][0]))
        relation = self.oqag.number2string(int(qa_num[0][1]))
        location = self.oqag.number2string(int(qa_num[0][2]))
        answer = self.oqag.number2string(int(qa_num[0][3]))

        assert answer == "<answer>"

        qa_sem = [obj, relation, location]
        logging.info("The numpy array has been converted to a semantic qa!")

        return qa_sem

    def sample(self):
        """Sample a state."""
        logging.debug(f"Sampling a state from the {self.space_type} space ...")
        self.M_e.forget_all()
        self.M_s.forget_all()

        me_max = self.M_e.capacity
        ms_max = self.M_s.capacity

        if self.space_type in [
            "episodic_memory_manage",
            "semantic_memory_manage",
            "episodic_to_semantic",
        ]:
            me_max += 1
            ms_max += 1

        if self.M_e.capacity > 0:
            self.oqag.reset()
            for _ in range(random.randint(0, self.M_e.capacity)):
                ob, _ = self.oqag.generate(generate_qa=False)
                mem_epi = self.M_e.ob2epi(ob)
                self.M_e.add(mem_epi)
            qa = self.oqag.generate_question_answer()
            qa_epi = qa

        if self.M_s.capacity > 0:
            self.oqag.reset()
            for _ in range(random.randint(0, self.M_s.capacity)):
                ob, _ = self.oqag.generate(generate_qa=False)
                mem_sem = self.M_s.ob2sem(ob)
                self.M_s.add(mem_sem)
            qa = self.oqag.generate_question_answer()
            qa_sem = self.M_s.eq2sq(qa)

        if self.space_type == "episodic_memory_manage":
            state_numeric = self.episodic_memory_system_to_numbers(self.M_e, me_max)

            return state_numeric

        elif self.space_type == "episodic_question_answer":
            state_numeric_1 = self.episodic_memory_system_to_numbers(self.M_e, me_max)
            state_numeric_2 = self.episodic_question_answer_to_numbers(qa_epi)

            state_numeric = np.concatenate([state_numeric_1, state_numeric_2])

            return state_numeric

        elif self.space_type == "semantic_memory_manage":
            state_numeric = self.semantic_memory_system_to_numbers(
                self.M_s, ms_max, pad=False
            )

            return state_numeric

        elif self.space_type == "semantic_question_answer":
            state_numeric_1 = self.semantic_memory_system_to_numbers(self.M_s, ms_max)
            state_numeric_2 = self.semantic_question_answer_to_numbers(qa_sem)

            state_numeric = np.concatenate([state_numeric_1, state_numeric_2])

            return state_numeric

        elif self.space_type == "episodic_to_semantic":
            state_numeric = self.episodic_memory_system_to_numbers(self.M_e, me_max)

            return state_numeric

        elif self.space_type == "episodic_semantic_question_answer":
            state_numeric_1 = self.episodic_memory_system_to_numbers(self.M_e, me_max)
            state_numeric_2 = self.semantic_memory_system_to_numbers(
                self.M_s, ms_max, pad=True
            )
            state_numeric_3 = self.episodic_question_answer_to_numbers(qa_epi)

            state_numeric = np.concatenate(
                [state_numeric_1, state_numeric_2, state_numeric_3]
            )

            return state_numeric

        raise ValueError

    def is_correct_memory_system(self, memory_type: str, entries: list) -> bool:
        """Check if a given list is a legit memory system.

        Args
        ----
        memory_type: either "episodic" or "semantic"
        entries: memories

        Returns
        -------
        True or False

        """
        logging.debug("Checking if given memories match the memory system ...")
        assert memory_type in ["episodic", "semantic"]
        if not isinstance(entries, list):
            return False

        for entry in entries:
            if len(entry) != 4:
                return False
            if Memory.remove_name(entry[0]) not in self.oqag.heads:
                return False
            if entry[1] not in self.oqag.relations:
                return False
            if Memory.remove_name(entry[2]) not in self.oqag.tails:
                return False
            if memory_type == "episodic":
                if not (0 <= entry[3] <= MAX_INT_32):
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
        logging.info("Given memories match the memory system!")

        return True

    def is_correct_question_answer(self, memory_type: str, qa: list) -> bool:
        """Check if a given question-answer is legit.

        Args
        ----
        memory_type: either "episodic" or "semantic"
        qa: question-answer

        Returns
        -------
        True or False

        """
        logging.debug("Checking if given qa is legit ...")
        if not isinstance(qa, list):
            return False

        if len(qa) != 3:
            return False

        if Memory.remove_name(qa[0]) not in self.oqag.heads:
            return False
        if qa[1] not in self.oqag.relations:
            return False
        if Memory.remove_name(qa[2]) not in self.oqag.tails:
            return False
        if memory_type == "episodic":
            if qa[0].split()[0].split("'s")[0] not in self.oqag.names:
                return False
            if qa[2].split()[0].split("'s")[0] not in self.oqag.names:
                return False

        logging.info("The given qa is legit!")

        return True

    def determine_space_qa(self, state_numeric: np.ndarray) -> bool:
        """Determine if the given numeric state is a qa or not.

        Args
        ----
        state_numeric: numpy array with rows and columns

        Returns
        -------
        True or False

        """
        assert state_numeric.shape[1] in [4, 6]

        if state_numeric[-1][-1] == self.oqag.string2number("<answer>"):
            is_qa = True
        else:
            is_qa = False

        return is_qa

    def contains(self, x):
        if not isinstance(x, np.ndarray):
            return False

        if len(x) > self.M_e.capacity + self.M_s.capacity + 1:
            return False

        is_qa = self.determine_space_qa(x)

        if x.shape[1] == 4:
            if is_qa:
                memories = x[:-1, :]
                qa = x[-1, :]

                if self.oqag.number2string(int(qa[0])) not in self.oqag.heads:
                    return False
                if self.oqag.number2string(int(qa[1])) not in self.oqag.relations:
                    return False
                if self.oqag.number2string(int(qa[2])) not in self.oqag.tails:
                    return False
                if self.oqag.number2string(int(qa[3])) != "<answer>":
                    return False
            else:
                memories = x

            for row in memories:
                obj = self.oqag.number2string(int(row[0]))
                relation = self.oqag.number2string(int(row[1]))
                location = self.oqag.number2string(int(row[2]))
                time_or_general = row[3]

                if obj not in self.oqag.heads + ["<pad>"]:
                    return False
                if relation not in self.oqag.relations + ["<pad>"]:
                    return False
                if location not in self.oqag.tails + ["<pad>"]:
                    return False
                if time_or_general != "<pad>" and not (
                    0 <= time_or_general <= MAX_INT_32
                ):
                    return False

        elif x.shape[1] == 6:
            if is_qa:
                memories = x[:-1, :]
                qa = x[-1, :]

                if self.oqag.number2string(int(qa[0])) not in self.oqag.names + ["<pad>"]:
                    return False
                if self.oqag.number2string(int(qa[1])) not in self.oqag.heads:
                    return False
                if self.oqag.number2string(int(qa[2])) not in self.oqag.relations:
                    return False
                if self.oqag.number2string(int(qa[3])) not in self.oqag.names + ["<pad>"]:
                    return False
                if self.oqag.number2string(int(qa[4])) not in self.oqag.tails:
                    return False
                if self.oqag.number2string(int(qa[5])) != "<answer>":
                    return False
            else:
                memories = x

            for row in memories:
                name1 = self.oqag.number2string(int(row[0]))
                obj = self.oqag.number2string(int(row[1]))
                relation = self.oqag.number2string(int(row[2]))
                name2 = self.oqag.number2string(int(row[3]))
                location = self.oqag.number2string(int(row[4]))
                time_or_general = row[5]

                if name1 not in self.oqag.names + ["<pad>"]:
                    return False
                if obj not in self.oqag.heads + ["<pad>"]:
                    return False
                if relation not in self.oqag.relations + ["<pad>"]:
                    return False
                if name2 not in self.oqag.names + ["<pad>"]:
                    return False
                if location not in self.oqag.tails + ["<pad>"]:
                    return False
                if time_or_general != "<pad>" and not (
                    0 <= time_or_general <= MAX_INT_32
                ):
                    return False

        else:
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
