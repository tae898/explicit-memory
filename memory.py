import os
import logging
import random
from pprint import pformat
from utils import read_json

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NO_HEADS_FOUND = 0
BOTH_HEAD_AND_TAIL = 2
ONLY_HEAD_NOT_TAIL = 1


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

    def __repr__(self):

        return pformat(vars(self), indent=4, width=1)

    def forget(self, mem: list):
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

    @property
    def is_empty(self) -> bool:
        """Return true if full."""
        return len(self.entries) == 0

    @property
    def is_full(self) -> bool:
        """Return true if full."""
        assert len(self.entries) <= self.capacity + 1
        return len(self.entries) == self.capacity or (
            len(self.entries) == self.capacity + 1
        )

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

        logging.info(f"{len(duplicates)} duplicates were found!")
        if len(duplicates) == 0:
            logging.debug("no duplicates were found!")
            return None
        else:
            return duplicates

    def forget_random(self) -> None:
        """Forget a memory in the memory system in a uniform distribution manner."""
        logging.warning("forgetting a random memory using a uniform distribution ...")
        mem = random.choice(self.entries)
        self.forget(mem)

    def answer_random(self, question: list) -> int:
        """Answer the question with a uniform-randomly chosen memory.

        Args
        ----
        question: a triple (i.e., (head, relation, tail))

        Returns
        -------
        reward: -1 if no heads were found, +1 if the retrieved memory matches both head
            and tail, and 0 if the retrieved memory only matches the head, not the tail.

        """
        logging.debug("answering a question with a uniform-random memory ...")
        assert len(question) == 3
        if self.is_empty:
            return NO_HEADS_FOUND

        query_head = question[0]
        query_tail = question[2]

        mem = random.choice(self.entries)
        head = mem[0]
        tail = mem[2]

        if head != query_head:
            logging.info("The retrieved memory is not correct for the question!")
            return NO_HEADS_FOUND
        else:
            if tail == query_tail:
                logging.info(f"retrieved the relevant memory {mem}!")
                return BOTH_HEAD_AND_TAIL
            else:
                logging.info("The object was there in the past, not now!")
                return ONLY_HEAD_NOT_TAIL

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

    def answer_newest(self, question: list) -> int:
        """Answer the question with the latest relevant memory.

        If object X was found at Y and then later on found Z, then this strategy answers
        Z, instead of Y.

        Args
        ----
        question: a triple (i.e., (head, relation, tail))

        Returns
        -------
        reward: -1 if no heads were found, +1 if the retrieved memory matches both head and
            tail, and 0 if the retrieved memory only matches the head, not the tail.

        """
        assert len(question) == 3
        logging.debug("answering a question with a random memory ...")
        if self.is_empty:
            return NO_HEADS_FOUND

        query_head = question[0]
        query_tail = question[2]
        duplicates = self.get_duplicate_heads(query_head, self.entries)
        if duplicates is None:
            logging.info("no relevant memories were found in the entries.")
            return NO_HEADS_FOUND
        else:
            logging.info(
                f"{len(duplicates)} relevant memories were found in the entries!"
            )
            mem = self.get_latest_memory(duplicates)
            tail = mem[2]
            if tail == query_tail:
                logging.info(
                    f"The latest strategy retrieved the relevant memory {mem}!"
                )
                return BOTH_HEAD_AND_TAIL
            else:
                logging.info("The object was there in the past, not now!")
                return ONLY_HEAD_NOT_TAIL

    def ob2epi(self, ob: list):
        """Turn an observation into an episodic memory.

        At the moment, an observation is the same as an episodic memory for
            simplification.

        Args
        ----
        ob: An observation in a quadruple format
            (i.e., (head, relation, tail, timestamp))

        """
        assert len(ob) == 4
        logging.debug(f"Turning an observation {ob} into a semantic memory ...")
        mem_epi = ob
        logging.info(f"Observation {ob} is now a episodic memory {mem_epi}")

        return mem_epi

    def get_similar(self, entries: list = None):
        """Find N episodic memories that can be compressed into one semantic.

        At the moment, this is simply done by matching string values. In the end, an
        RL agent has to learn this by itself.

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
        semantic_possibles = [[e.split()[-1] for e in entry[:-1]] for entry in entries]
        # "_" is to allow hashing.
        semantic_possibles = ["_".join(elem) for elem in semantic_possibles]

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
            semantic_memory = max_key.split("_")
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

    def pretrain_semantic(self, pretrain_path: str) -> int:
        """Pretrain the semantic memory system from ConceptNet.

        Args
        ----
        pretrain_path: the path to the pretrained semantic memory.

        Returns
        -------
        free_space: free space that can be added to the episodic memory system.

        """
        logging.debug("Populating the semantic memory system with prior knowledge ...")
        pretrained = read_json(pretrain_path)
        for head, relation_head in pretrained.items():
            if self.is_full:
                break

            relation = list(relation_head.keys())[0]
            tail = relation_head[relation][0]
            # num_generalized_memories is a possible maximum value.
            mem = [head, relation, tail, 9223372036854775807]
            logging.debug(f"adding {mem} to the semantic memory system ...")
            self.add(mem)

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
        logging.info(f"{mem} is the oldest memory in the entries.")

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
        reward: -1 if no heads were found, +1 if the retrieved memory matches both head and
            tail, and 0 if the retrieved memory only matches the head, not the tail.

        """
        assert len(question) == 3
        logging.debug("answering a question with a uniform-randomly chosen memory ...")
        if self.is_empty:
            return NO_HEADS_FOUND

        query_head = question[0]
        query_tail = question[2]
        duplicates = self.get_duplicate_heads(query_head, self.entries)
        if duplicates is None:
            logging.info("no relevant memories were found in the entries.")
            return NO_HEADS_FOUND
        else:
            logging.info(
                f"{len(duplicates)} relevant memories were found in the entries!"
            )
            mem = self.get_strongest_memory(duplicates)
            tail = mem[2]
            if tail == query_tail:
                logging.info(
                    f"STRONGEST strategy succesfully retrieved the relevant memory {mem}!"
                )
                return BOTH_HEAD_AND_TAIL
            else:
                logging.info("The object was there in the past, not now!")
                return ONLY_HEAD_NOT_TAIL

    def ob2sem(self, ob: list) -> list:
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
        head = ob[0].split()[-1]
        relation = ob[1]
        tail = ob[2].split()[-1]

        # 1 stands for the 1 generalized.
        mem_sem = [head, relation, tail, 1]
        logging.info(f"Observation {ob} is now a semantic memory {mem_sem}")

        return mem_sem

    def eq2sq(self, episodic_question) -> list:
        """Turn an episodic question to a semantic question.

        At the moment, this is simply done by removing the names from the head and the
        tail. In the end, an RL agent has to learn this by itself.

        Args
        ----
        question: An episodic question in a quadruple format
            (i.e., (head, relation, tail))

        """
        assert len(episodic_question) == 3
        logging.debug(
            f"Turning an episodic question {episodic_question} into a semantic "
            f"question ..."
        )
        # split to remove the name
        head = episodic_question[0].split()[-1]
        relation = episodic_question[1]
        tail = episodic_question[2].split()[-1]

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

        entries = ["_".join(mem) for mem in entries]  # to make list hashable
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
        """Append a memory to the episodic memory system.

        Args
        ----
        mem: A memory in a quadruple format (i.e., (head, relation, tail,
            num_generalized_memories))

        """
        super().add(mem)
        self.clean_memories()
