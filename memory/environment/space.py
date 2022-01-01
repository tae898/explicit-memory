import logging
import os
import random
from pprint import pformat

import numpy as np
from gym.spaces import Space

from ..constants import MAX_INT_32
from ..memory import EpisodicMemory, Memory, SemanticMemory
from .generator import OQAGenerator

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": True,
        },
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
        limits: Limit the heads, tails per head, and the number of names. For example,
            this can be {"heads": 10, "tails": 1, "names" 10, "allow_spaces": True}

        """
        assert space_type in [
            "episodic_memory_manage",
            "episodic_question_answer",
            "semantic_memory_manage",
            "semantic_question_answer",
            "episodic_to_semantic",
            "episodic_semantic_question_answer",
        ]
        self.dtype = np.float32
        if space_type == "episodic_memory_manage":
            assert capacity["episodic"] > 0 and capacity["semantic"] == 0
            self.shape = (capacity["episodic"] + 1, 6)

        elif space_type == "episodic_question_answer":
            assert capacity["episodic"] > 0 and capacity["semantic"] == 0
            self.shape = (capacity["episodic"] + 1, 6)

        elif space_type == "semantic_memory_manage":
            assert capacity["episodic"] == 0 and capacity["semantic"] > 0
            self.shape = (capacity["semantic"] + 1, 4)

        elif space_type == "semantic_question_answer":
            assert capacity["episodic"] == 0 and capacity["semantic"] > 0
            self.shape = (capacity["semantic"] + 1, 4)

        elif space_type == "episodic_to_semantic":
            assert capacity["episodic"] > 0 and capacity["semantic"] > 0
            raise NotImplementedError

        elif space_type == "episodic_semantic_question_answer":
            assert capacity["episodic"] > 0 and capacity["semantic"] > 0
            raise NotImplementedError

        else:
            raise ValueError

        self.space_type = space_type
        self.capacity = capacity

        self.oqag = OQAGenerator(
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            limits=limits,
        )
        self.M_e = EpisodicMemory(self.capacity["episodic"])
        self.M_s = SemanticMemory(self.capacity["semantic"])

        super().__init__(shape=self.shape, dtype=self.dtype)

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
        assert state_numeric.shape[1] in [4, 6]
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
                else:
                    entries.append([obj, relation, location, num_general])

        else:
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

                else:
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
        qa_num: [name1, obj, relation, name2, <MASK>, <pad>]

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
        qa_num[0][4] = self.oqag.string2number("<MASK>")
        qa_num[0][5] = self.oqag.string2number("<pad>")

        logging.debug("The numpy array has been converted to an episodic qa!")

        return qa_num

    def numbers_to_episodic_question_answer(self, qa_num: np.ndarray) -> list:
        """Convert a given numpy array to an episodic qa.

        Args
        ----
        qa_num: [name1, obj, relation, name2, <MASK>, <pad>]


        Returns
        -------
        qa_epi: episodic question and answer, [head, relation, ?]

        """
        logging.debug("Converting the numpy array to an episodic qa ...")

        assert qa_num.shape == (1, 6)

        name1 = self.oqag.number2string(int(qa_num[0][0]))
        obj = self.oqag.number2string(int(qa_num[0][1]))
        relation = self.oqag.number2string(int(qa_num[0][2]))
        name2 = self.oqag.number2string(int(qa_num[0][3]))
        mask = self.oqag.number2string(int(qa_num[0][4]))
        pad = self.oqag.number2string(int(qa_num[0][5]))

        assert mask == "<MASK>"
        assert pad == "<pad>"
        assert name1 == name2

        qa_epi = [f"{name1}'s {obj}", relation, f"{name2}'s location"]

        logging.info("The numpy array has been converted to an episodic qa!")

        return qa_epi

    def semantic_question_answer_to_numbers(
        self, qa_sem: list, pad: bool = False
    ) -> np.ndarray:
        """Convert a given semantic qa to numbers.

        This conversion is necessary so that we can use a computational model.

        Args
        ----
        qa_sem: semantic question and answer, [head, relation, tail]

        Returns
        -------
        qa_num: [obj, relation, <MASK>, <pad>]

        """
        logging.debug("Converting a given semantic qa into a numpy array ...")

        if pad:
            NUM_COLUMNS = 6

        else:
            NUM_COLUMNS = 4

        qa_num = np.zeros((1, NUM_COLUMNS), dtype=np.float32)
        head = qa_sem[0]
        relation = qa_sem[1]
        tail = qa_sem[2]

        if pad:
            qa_num[0][0] = self.oqag.string2number("<pad>")
            qa_num[0][1] = self.oqag.string2number(head)
            qa_num[0][2] = self.oqag.string2number(relation)
            qa_num[0][3] = self.oqag.string2number("<pad>")
            qa_num[0][4] = self.oqag.string2number("<MASK>")
            qa_num[0][5] = self.oqag.string2number("<pad>")

        else:
            qa_num[0][0] = self.oqag.string2number(head)
            qa_num[0][1] = self.oqag.string2number(relation)
            qa_num[0][2] = self.oqag.string2number("<MASK>")
            qa_num[0][3] = self.oqag.string2number("<pad>")

        logging.debug("The given semantic qa has been converted to a numpy array!")

        return qa_num

    def numbers_to_semantic_question_answer(self, qa_num: np.ndarray) -> list:
        """Convert a given numpy array to an semantic qa.

        Args
        ----
        qa_num: [obj, relation, <MASK>, <pad>]


        Returns
        -------
        qa_sem: semantic question and answer, [head, relation, ?]

        """
        logging.debug("Converting the numpy array to a semantic qa ...")
        assert qa_num.shape in [(1, 4), (1, 6)]

        if qa_num.shape == (1, 6):
            name1 = self.oqag.number2string(int(qa_num[0][0]))
            obj = self.oqag.number2string(int(qa_num[0][1]))
            relation = self.oqag.number2string(int(qa_num[0][2]))
            name2 = self.oqag.number2string(int(qa_num[0][3]))
            mask = self.oqag.number2string(int(qa_num[0][4]))
            pad = self.oqag.number2string(int(qa_num[0][5]))

            assert name1 == name2 == pad == "<pad>"
        else:
            obj = self.oqag.number2string(int(qa_num[0][0]))
            relation = self.oqag.number2string(int(qa_num[0][1]))
            mask = self.oqag.number2string(int(qa_num[0][2]))
            pad = self.oqag.number2string(int(qa_num[0][3]))

        assert mask == "<MASK>"

        qa_sem = [obj, relation, "?"]
        logging.info("The numpy array has been converted to a semantic qa!")

        return qa_sem

    def sample(self):
        """Sample a state."""
        logging.debug(f"Sampling a state from the {self.space_type} space ...")

        if self.space_type == "episodic_memory_manage":
            assert self.M_e.capacity > 0
            self.oqag.reset()
            self.M_e.forget_all()
            for _ in range(self.M_e.capacity + 1):
                ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
                mem_epi = self.M_e.ob2epi(ob)
                self.M_e.add(mem_epi)

            state_numeric = self.episodic_memory_system_to_numbers(
                self.M_e, self.M_e.capacity + 1
            )

            return state_numeric

        elif self.space_type == "episodic_question_answer":
            assert self.M_e.capacity > 0
            self.oqag.reset()
            self.M_e.forget_all()
            # for _ in range(random.randint(0, self.M_e.capacity)):
            for _ in range(self.M_e.capacity):
                ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
                mem_epi = self.M_e.ob2epi(ob)
                self.M_e.add(mem_epi)

            state_numeric_1 = self.episodic_memory_system_to_numbers(
                self.M_e, self.M_e.capacity
            )
            state_numeric_2 = self.episodic_question_answer_to_numbers(qa_epi)

            state_numeric = np.concatenate([state_numeric_1, state_numeric_2])

            return state_numeric

        elif self.space_type == "semantic_memory_manage":
            assert self.M_s.capacity > 0
            self.oqag.reset()
            self.M_s.forget_all()
            for _ in range(self.M_s.capacity + 1):
                ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
                mem_sem = self.M_s.ob2sem(ob)
                self.M_s.add(mem_sem)
            qa_sem = self.M_s.eq2sq(qa_epi)
            state_numeric = self.semantic_memory_system_to_numbers(
                self.M_s, self.M_s.capacity + 1, pad=False
            )

            return state_numeric

        elif self.space_type == "semantic_question_answer":
            assert self.M_s.capacity > 0
            self.oqag.reset()
            self.M_s.forget_all()
            # for _ in range(random.randint(0, self.M_s.capacity)):
            for _ in range(self.M_s.capacity):
                ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
                mem_sem = self.M_s.ob2sem(ob)
                self.M_s.add(mem_sem)
            qa_sem = self.M_s.eq2sq(qa_epi)

            state_numeric_1 = self.semantic_memory_system_to_numbers(
                self.M_s, self.M_s.capacity, pad=False
            )
            state_numeric_2 = self.semantic_question_answer_to_numbers(qa_sem)

            state_numeric = np.concatenate([state_numeric_1, state_numeric_2])

            return state_numeric

        # elif self.space_type == "episodic_to_semantic":
        #     assert self.M_e.capacity > 0 and self.M_s.capacity > 0
        #     self.oqag.reset()
        #     self.M_e.forget_all()
        #     for _ in range(self.M_e.capacity + 1):
        #         ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
        #         mem_epi = self.M_e.ob2epi(ob)
        #         self.M_e.add(mem_epi)

        #     state_numeric = self.episodic_memory_system_to_numbers(
        #         self.M_e, self.M_e.capacity + 1
        #     )

        #     return state_numeric

        # elif self.space_type == "episodic_semantic_question_answer":
        #     assert self.M_e.capacity > 0 and self.M_s.capacity > 0
        #     self.oqag.reset()
        #     self.M_e.forget_all()
        #     self.M_s.forget_all()

        #     for _ in range(random.randint(0, self.M_e.capacity)):
        #         ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
        #         mem_epi = self.M_e.ob2epi(ob)
        #         self.M_e.add(mem_epi)
        #     state_numeric_e = self.episodic_memory_system_to_numbers(
        #         self.M_e, self.M_e.capacity
        #     )
        #     state_numeric_eq = self.episodic_question_answer_to_numbers(qa_epi)

        #     for _ in range(random.randint(0, self.M_s.capacity)):
        #         ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
        #         mem_sem = self.M_s.ob2sem(ob)
        #         self.M_s.add(mem_sem)
        #     qa_sem = self.M_s.eq2sq(qa_epi)
        #     state_numeric_s = self.semantic_memory_system_to_numbers(
        #         self.M_s, self.M_s.capacity, pad=False
        #     )
        #     state_numeric_sq = self.semantic_question_answer_to_numbers(qa_sem)

        #     state_numeric = np.concatenate(
        #         [state_numeric_e, state_numeric_eq, state_numeric_s,state_numeric_sq]
        #     )

        #     return state_numeric

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
                if entry[0].split()[0].split("'s")[0] not in self.oqag.names:
                    return False
                if entry[2].split()[0].split("'s")[0] not in self.oqag.names:
                    return False
            else:
                if not (1 <= entry[3] <= MAX_INT_32):
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

        if state_numeric[-1][-1] == self.oqag.string2number("<MASK>"):
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
                if self.oqag.number2string(int(qa[3])) != "<MASK>":
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

                if self.oqag.number2string(int(qa[0])) not in self.oqag.names + [
                    "<pad>"
                ]:
                    return False
                if self.oqag.number2string(int(qa[1])) not in self.oqag.heads:
                    return False
                if self.oqag.number2string(int(qa[2])) not in self.oqag.relations:
                    return False
                if self.oqag.number2string(int(qa[3])) not in self.oqag.names + [
                    "<pad>"
                ]:
                    return False
                if self.oqag.number2string(int(qa[4])) not in self.oqag.tails:
                    return False
                if self.oqag.number2string(int(qa[5])) != "<MASK>":
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
