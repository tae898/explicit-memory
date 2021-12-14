import logging
import os
import random
from pprint import pformat
from typing import List, Tuple

from .constants import CORRECT, WRONG
from .utils import read_json

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Memory:
    """Memory (episodic or semantic) class"""

    def __init__(self, memory_type: str, capacity: int) -> None:
        """

        Args
        ----
        memory_type: either episodic or semantic.
        capacity: memory capacity

        """
        logging.debug(
            f"instantiating a {memory_type} memory object with size {capacity} ..."
        )

        assert memory_type in ["episodic", "semantic"]
        self.type = memory_type
        self.entries = []
        self.capacity = capacity
        self._frozen = False

        logging.debug(f"{memory_type} memory object with size {capacity} instantiated!")

    def __eq__(self, other):
        eps = 0.01
        if self.memory_type != other.memory_type:
            return False
        if self.capacity != other.capacity:
            return False
        if self._frozen != other._frozen:
            return False

        if len(self.entries) != len(other.entries):
            return False

        for se, oe in zip(self.entries, other.entries):
            if se[0] != oe[0]:
                return False
            if se[1] != oe[1]:
                return False
            if se[2] != oe[2]:
                return False
            if abs(se[3] - oe[3]) > eps:
                return False

        return True

    def __repr__(self):

        return pformat(vars(self), indent=4, width=1)

    def is_answerable(self, question: list) -> bool:
        """Check if a relevant memory is in the system to answer the given question.

        Args
        ----
        question: a triple (i.e., (head, relation, tail))

        """
        logging.debug(
            f"Check if a relevant memory is in the system to answer the "
            f"given question: {question} ..."
        )

        question_head = question[0]
        question_relation = question[1]

        for mem in self.entries:
            head = mem[0]
            assert len(question_head.split()) == len(
                head.split()
            ), f"{question_head.split()}, {head.split()}"
            relation = mem[1]

            if head == question_head and relation == question_relation:
                logging.info("There is a relevant memory to answer the question!")
                return True

        logging.info("There is NOT a relevant memory to answer the question!")

        return False

    def forget(self, mem: list) -> None:
        """forget the given memory.

        Args
        ----
        mem: A memory in a quadruple format (i.e., (head, relation, tail, timestamp)
            for episodic and (head, relation, tail, num_generalized_memories) for semantic)

        """
        if self._frozen:
            error_msg = "The memory system is frozen!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        if mem not in self.entries:
            error_msg = f"{mem} is not in the memory system!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Forgetting {mem} ...")
        self.entries.remove(mem)
        logging.info(f"{mem} forgotten!")

    def forget_all(self) -> None:
        """Forget everything in the memory system!"""
        logging.warning("EVERYTHING IN THE MEMORY SYSTEM WILL BE FORGOTTEN!")
        self.entries = []

    @property
    def is_empty(self) -> bool:
        """Return true if full."""
        return len(self.entries) == 0

    @property
    def is_full(self) -> bool:
        """Return true if full."""
        assert len(self.entries) <= self.capacity + 1

        return len(self.entries) == self.capacity

    @property
    def is_kinda_full(self) -> bool:
        """Return if one is dangling."""
        assert len(self.entries) <= self.capacity + 1

        return len(self.entries) == self.capacity + 1

    @property
    def is_frozen(self):
        """Is frozen?"""
        return self._frozen

    @property
    def size(self) -> int:
        """Get the size (number of filled entries) of the memory system."""
        return len(self.entries)

    def freeze(self):
        """Freeze the memory so that nothing can be added / deleted."""
        self._frozen = True

    def unfreeze(self):
        """Unfreeze the memory so that something can be added / deleted."""
        self._frozen = False

    def get_duplicate_heads(self, head: str, entries: list = None) -> list:
        """Find if there are duplicate heads and return its index.

        At the moment, this is simply done by matching string values. In the end, an
        RL agent has to learn this by itself.

        Args
        ----
        head: (e.g., Tae's laptop)

        Returns
        -------
        duplicates: a list of memories. Every memory in this list has the same head as
            that of the observation (i.e., (head, relation, tail,
            timestamp/num_generalized_memories)). None is returned when no duplicate heads
            were found.

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        logging.debug("finding if duplicate heads exist ...")
        duplicates = []
        for mem in entries:
            if mem[0] == head:
                logging.info(f"{mem} has the same head {head} !!!")
                duplicates.append(mem)

        if len(duplicates) == 0:
            logging.debug("no duplicates were found!")
            return None

        logging.info(f"{len(duplicates)} duplicates were found!")

        return duplicates

    def forget_random(self) -> None:
        """Forget a memory in the memory system in a uniform distribution manner."""
        logging.warning("forgetting a random memory using a uniform distribution ...")
        mem = random.choice(self.entries)
        self.forget(mem)

    @staticmethod
    def remove_name(entity: str) -> str:
        """Remove name from the entity.

        Args
        ----
        entity: e.g., Tae's laptop

        """
        return entity.split()[-1]

    @staticmethod
    def is_question_valid(question) -> bool:
        """Check if the given question is valid.

        Args
        ----
        question: a triple (i.e., (head, relation, tail))

        Returns
        -------
        valid: True or not.

        """
        logging.debug(f"Checking if the question {question} is valid ..")
        if len(question) == 3:
            logging.info(f"{question} is a valid question.")
            return True
        else:
            logging.info(f"{question} is NOT a valid question.")
            return False

    def answer_random(self, question: list) -> int:
        """Answer the question with a uniform-randomly chosen memory.

        Args
        ----
        question: a triple (i.e., (head, relation, tail))

        Returns
        -------
        reward: CORRECT or WRONG
        pred: prediction
        answer: correct answer

        """
        if not self.is_question_valid(question):
            raise ValueError
        logging.debug(
            "answering the question with a uniform-randomly retrieved memory ..."
        )
        correct_answer = self.remove_name(question[2])

        if self.is_empty:
            pred = None
            return WRONG, pred, correct_answer

        pred = self.remove_name(random.choice(self.entries)[2])

        if pred == correct_answer:
            reward = CORRECT
        else:
            reward = WRONG

        logging.info(
            f"pred: {pred}, correct answer: {correct_answer}. Reward: {reward}"
        )

        return reward, pred, correct_answer

    def add(self, mem: list):
        """Append a memory to the memory system.

        Args
        ----
        mem: A memory in a quadruple format (i.e., (head, relation, tail,
            timestamp/num_generalized_memories))

        """
        assert len(mem) == 4
        if self._frozen:
            error_msg = "The memory system is frozen!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # +1 is to account for the incoming observation which can possibly be a
        # memory
        if len(self.entries) >= self.capacity + 1:
            error_msg = "memory is full! can't add more."
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Adding a new memory entry {mem} ...")
        self.entries.append(mem)
        logging.info(
            f"memory entry {mem} added. Now there are in total of "
            f"{len(self.entries)} memories!"
        )

    def increase_capacity(self, increase):
        """Increase the capacity.

        Args
        ----
        increase: the amount of entries to increase.

        """
        assert isinstance(increase, int) and (not self.is_frozen)
        logging.debug(f"Increasing the memory capacity by {increase} ...")
        self.capacity += increase
        logging.info(
            f"The memory capacity has been increased by {increase} and now it's "
            f"{self.capacity}!"
        )

    def decrease_capacity(self, decrease):
        """decrease the capacity.

        Args
        ----
        decrease: the amount of entries to decrease.

        """
        assert (
            isinstance(decrease, int)
            and (self.capacity - decrease >= 0)
            and (not self.is_frozen)
        )
        logging.debug(f"Decreasing the memory capacity by {decrease} ...")
        self.capacity -= decrease
        logging.info(
            f"The memory capacity has been decreased by {decrease} and now it's "
            f"{self.capacity}!"
        )


class EpisodicMemory(Memory):
    """Episodic memory class."""

    def __init__(self, capacity: int) -> None:
        super().__init__("episodic", capacity)

    def get_oldest_memory(self, entries: list = None) -> list:
        """Get the oldest memory in the episodic memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them. In the end, an RL agent has to learn this by itself.

        Returns
        -------
        mem: the oldest memory in a quadruple format (i.e., (head, relation, tail,
            timestamp))

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # The last element [-1] of a memory is timestamp
        mem = sorted(entries, key=lambda x: x[-1])[0]
        assert len(mem) == 4
        logging.info(f"{mem} is the oldest memory in the entries.")

        return mem

    def get_latest_memory(self, entries: list = None) -> list:
        """Get the latest memory in the episodic memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them. In the end, an RL agent has to learn this by itself.

        Returns
        -------
        mem: the latest memory in a quadruple format (i.e., (head, relation, tail,
            timestamp))

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        mem = sorted(entries, key=lambda x: x[-1])[-1]
        assert len(mem) == 4
        logging.info(f"{mem} is the latest memory in the entries.")

        return mem

    def forget_oldest(self) -> None:
        """Forget the oldest entry in the memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them. In the end, an RL agent has to learn this by itself.

        """
        logging.debug("forgetting the oldest memory (FIFO)...")

        mem = self.get_oldest_memory()
        self.forget(mem)

    def answer_latest(self, question: list) -> int:
        """Answer the question with the latest relevant memory.

        If object X was found at Y and then later on found Z, then this strategy answers
        Z, instead of Y.

        Args
        ----
        question: a triple (i.e., (head, relation, tail))

        Returns
        -------
        reward: CORRECT or WRONG
        pred: prediction
        answer: correct answer

        """
        if not self.is_question_valid(question):
            raise ValueError
        logging.debug("answering a question with the answer_latest policy ...")
        correct_answer = self.remove_name(question[2])

        if self.is_empty:
            pred = None
            return WRONG, pred, correct_answer

        query_head = question[0]
        duplicates = self.get_duplicate_heads(query_head, self.entries)
        if duplicates is None:
            logging.info("no relevant memories found.")
            pred = None

            reward = WRONG

        else:
            logging.info(
                f"{len(duplicates)} relevant memories were found in the entries!"
            )
            mem = self.get_latest_memory(duplicates)
            pred = self.remove_name(mem[2])

            if pred == correct_answer:
                reward = CORRECT
            else:
                reward = WRONG

        logging.info(
            f"pred: {pred}, correct answer: {correct_answer}. Reward: {reward}"
        )

        return reward, pred, correct_answer

    @staticmethod
    def ob2epi(ob: list):
        """Turn an observation into an episodic memory.

        At the moment, an observation is the same as an episodic memory for
        simplification.

        Args
        ----
        ob: An observation in a quadruple format
            (i.e., (head, relation, tail, timestamp))

        """
        assert len(ob) == 4
        logging.debug(f"Turning an observation {ob} into a episodic memory ...")
        mem_epi = ob
        logging.info(f"Observation {ob} is now a episodic memory {mem_epi}")

        return mem_epi

    @staticmethod
    def remove_timestamp(entry: list) -> list:
        """Remove the timestamp from a given observation/episodic memory.

        Args
        ----
        entry: An observation / episodic memory in a quadruple format
            (i.e., (head, relation, tail, timestamp))

        Returns
        -------
        entry_without_timestamp: i.e., (head, relation, tail)

        """
        assert len(entry) == 4
        logging.debug(f"Removing timestamp from {entry} ...")
        entry_without_timestamp = entry[:-1]
        logging.info(f"Timestamp is removed from {entry}: {entry_without_timestamp}")

        return entry_without_timestamp

    @staticmethod
    def split_name_entity(name_entity: str) -> Tuple[str, str]:
        """Separate name and entity from the given string.

        Args
        ----
        name_entity: e.g., "Tae's laptop"

        Returns
        -------
        name: e.g., Tae
        entity: e.g., laptop

        """
        logging.debug(f"spliting name and entity from {name_entity}")
        splitted = name_entity.split()
        assert len(splitted) == 2 and "'" in splitted[0]
        name = splitted[0].split("'")[0]
        entity = splitted[1]

        return name, entity

    def get_similar(self, entries: list = None):
        """Find N episodic memories that can be compressed into one semantic.

        At the moment, this is simply done by matching string values. In the end, a
        neural network has to learn this by itself (e.g., symbolic knowledge graph
        compression).

        Returns
        -------
        episodic_memories: similar episodic memories
        semantic_memory: encoded (compressed) semantic memory in a quadruple format
            (i.e., (head, relation, tail, num_generalized_memories))

        """
        logging.debug("looking for episodic entries that can be compressed ...")
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # -1 removes the timestamps from the quadruples
        semantic_possibles = [
            [self.remove_name(e) for e in self.remove_timestamp(entry)]
            for entry in entries
        ]
        # "^" is to allow hashing.
        semantic_possibles = ["^".join(elem) for elem in semantic_possibles]

        def duplicates(mylist, item):
            return [i for i, x in enumerate(mylist) if x == item]

        semantic_possibles = dict(
            (x, duplicates(semantic_possibles, x)) for x in set(semantic_possibles)
        )

        if len(semantic_possibles) == len(entries):
            logging.info("no episodic memories found to be compressible.")
            return None, None
        elif len(semantic_possibles) < len(entries):
            logging.debug("some episodic memories found to be compressible.")

            max_key = max(semantic_possibles, key=lambda k: len(semantic_possibles[k]))
            indexes = semantic_possibles[max_key]

            episodic_memories = map(entries.__getitem__, indexes)
            episodic_memories = list(episodic_memories)
            # sort from the oldest to the latest
            episodic_memories = sorted(episodic_memories, key=lambda x: x[-1])
            semantic_memory = max_key.split("^")
            # num_generalized_memories is the number of compressed episodic memories.
            semantic_memory.append(len(indexes))
            assert (len(semantic_memory)) == 4
            for mem in episodic_memories:
                assert len(mem) == 4

            logging.info(
                f"{len(indexes)} episodic memories can be compressed "
                f"into one semantic memory: {semantic_memory}."
            )

            return episodic_memories, semantic_memory
        else:
            raise ValueError


class SemanticMemory(Memory):
    """Semantic memory class."""

    def __init__(self, capacity: int) -> None:
        super().__init__("semantic", capacity)

    def pretrain_semantic(
        self, semantic_knowledge_path: str, weighting_mode: str
    ) -> int:
        """Pretrain the semantic memory system from ConceptNet.

        Args
        ----
        semantic_knowledge_path: the path to the pretrained semantic memory.
        weighting_mode: "highest" chooses the one with the highest weight, "weighted"
            chooses all of them by weight, and null chooses every single one of them
            without weighting.

        Returns
        -------
        free_space: free space that can be added to the episodic memory system.

        """
        logging.debug(
            f"loading the semantic knowledge from {semantic_knowledge_path}..."
        )
        semantic_knowledge = read_json(semantic_knowledge_path)
        logging.info(
            f"semantic knowledge successfully loaded from {semantic_knowledge_path}!"
        )

        for head, relation_tails in semantic_knowledge.items():
            if self.is_full:
                break

            if weighting_mode == "weighted":
                for relation, tails in relation_tails.items():
                    for tail in tails:
                        mem = [head, relation, tail["tail"], tail["weight"]]

                        logging.debug(
                            f"weighting mode: {weighting_mode}: adding {mem} to the "
                            "semantic memory system ..."
                        )
                        self.add(mem)

            elif weighting_mode == "highest":
                for relation, tails in relation_tails.items():
                    tail = sorted(tails, key=lambda x: x["weight"])[-1]
                    mem = [head, relation, tail["tail"], tail["weight"]]

                    logging.debug(
                        f"weighting mode: {weighting_mode}: adding {mem} to the "
                        "semantic memory system ..."
                    )
                    self.add(mem)
            else:
                raise ValueError

        free_space = self.capacity - len(self.entries)
        self.decrease_capacity(free_space)
        self.freeze()
        logging.info("The semantic memory system is frozen!")
        logging.info(
            f"The semantic memory is pretrained and frozen. The remaining space "
            f"{free_space} will be returned. Now the capacity of the semantic memory "
            f"system is {self.capacity}"
        )

        return free_space

    def get_weakest_memory(self, entries: list = None) -> list:
        """Get the weakest memory in the semantic memory system system.

        At the moment, this is simply done by looking up the number of generalized
        episodic memories comparing them. In the end, an RL agent has to learn this
        by itself.

        Returns
        -------
        mem: the weakest memory in a quadruple format (i.e., (head, relation, tail,
            num_generalized_memories))

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # The last element [-1] of memory is num_generalized_memories.
        mem = sorted(entries, key=lambda x: x[-1])[0]
        logging.info(f"{mem} is the weakest memory in the entries.")

        return mem

    def get_strongest_memory(self, entries: list = None) -> list:
        """Get the strongest memory in the semantic memory system system.

        At the moment, this is simply done by looking up the number of generalized
        episodic memories comparing them. In the end, an RL agent has to learn this
        by itself.

        Returns
        -------
        mem: the strongest memory in a quadruple format (i.e., (head, relation, tail,
            num_generalized_memories))

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # The last element [-1] of memory is num_generalized_memories.
        mem = sorted(entries, key=lambda x: x[-1])[-1]
        logging.info(f"{mem} is the strongest memory in the entries.")

        return mem

    def forget_weakest(self) -> None:
        """Forget the weakest entry in the semantic memory system.

        At the moment, this is simply done by looking up the number of generalized
        episodic memories and comparing them. In the end, an RL agent has to learn this
        by itself.

        """
        logging.debug("forgetting the weakest memory ...")
        mem = self.get_weakest_memory()
        self.forget(mem)
        logging.info(f"{mem} is forgotten!")

    def answer_strongest(self, question: list) -> int:
        """Answer the question (Find the head that matches the question, and choose the
        strongest one among them).

        Args
        ----
        question: a triple (i.e., (head, relation, tail))

        Returns
        -------
        reward: CORRECT or WRONG
        pred: prediction
        answer: correct answer

        """
        if not self.is_question_valid(question):
            raise ValueError
        logging.debug("answering a question with the answer_strongest policy ...")
        correct_answer = self.remove_name(question[2])

        if self.is_empty:
            pred = None
            return WRONG, pred, correct_answer

        query_head = self.remove_name(question[0])
        duplicates = self.get_duplicate_heads(query_head, self.entries)
        if duplicates is None:
            logging.info("no relevant memories found.")
            pred = None

            reward = WRONG

        else:
            logging.info(
                f"{len(duplicates)} relevant memories were found in the entries!"
            )
            mem = self.get_strongest_memory(duplicates)
            pred = self.remove_name(mem[2])

            if pred == correct_answer:
                reward = CORRECT
            else:
                reward = WRONG

        logging.info(
            f"pred: {pred}, correct answer: {correct_answer}. Reward: {reward}"
        )

        return reward, pred, correct_answer

    @staticmethod
    def ob2sem(ob: list) -> list:
        """Turn an observation into a semantic memory.

        At the moment, this is simply done by removing the names from the head and the
        tail. In the end, an RL agent has to learn this by itself.

        Args
        ----
        ob: An observation in a quadruple format
            (i.e., (head, relation, tail, timestamp))

        """
        assert len(ob) == 4
        logging.debug(f"Turning an observation {ob} into a semantic memory ...")
        # split to remove the name
        head = Memory.remove_name(ob[0])
        relation = ob[1]
        tail = Memory.remove_name(ob[2])

        # 1 stands for the 1 generalized.
        mem_sem = [head, relation, tail, 1]
        logging.info(f"Observation {ob} is now a semantic memory {mem_sem}")

        return mem_sem

    @staticmethod
    def eq2sq(episodic_question) -> list:
        """Turn an episodic question to a semantic question.

        At the moment, this is simply done by removing the names from the head and the
        tail. In the end, an RL agent has to learn this by itself.

        Args
        ----
        question: An episodic question in a quadruple format
            (i.e., (head, relation, tail))

        """
        if not Memory.is_question_valid(episodic_question):
            raise ValueError

        logging.debug(
            f"Turning an episodic question {episodic_question} into a semantic "
            f"question ..."
        )
        # split to remove the name
        head = Memory.remove_name(episodic_question[0])
        relation = episodic_question[1]
        tail = Memory.remove_name(episodic_question[2])

        semantic_question = [head, relation, tail]
        logging.info(
            f"An episodic question {episodic_question} is now a semantic "
            f"question {semantic_question}"
        )
        return semantic_question

    def clean_memories(self) -> list:
        """Find if there are duplicate memories cuz they should be summed out.

        At the moment, this is simply done by matching string values. In the end, an
        RL agent has to learn this by itself.

        """
        logging.debug("finding if duplicate memories exist ...")

        entries = self.entries
        logging.debug(
            f"There are in total of {len(entries)} semantic memories before cleaning"
        )
        entries = [mem[:-1] for mem in entries]
        for mem in entries:
            assert len(mem) == 3

        entries = ["^".join(mem) for mem in entries]  # to make list hashable
        uniques = set(entries)

        def list_duplicates_of(seq, item):
            # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
            start_at = -1
            locs = []
            while True:
                try:
                    loc = seq.index(item, start_at + 1)
                except ValueError:
                    break
                else:
                    locs.append(loc)
                    start_at = loc
            return locs

        locs_all = [
            list_duplicates_of(entries, unique_entry) for unique_entry in uniques
        ]
        locs_all.sort(key=len)
        entries_cleaned = []
        for locs in locs_all:
            mem_sem = self.entries[locs[0]]
            mem_sem[-1] = sum([self.entries[loc][-1] for loc in locs])
            entries_cleaned.append(mem_sem)

        self.entries = entries_cleaned
        logging.debug(
            f"There are now in total of {len(self.entries)} semantic memories after cleaning"
        )

    def add(self, mem: list):
        """Append a memory to the semantic memory system.

        Args
        ----
        mem: A memory in a quadruple format (i.e., (head, relation, tail,
            num_generalized_memories))

        """
        super().add(mem)
        self.clean_memories()
