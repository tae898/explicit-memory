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
        mem: A memory in a quadruple format (i.e., (head, relation, tail, timestamp))

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

    def add(self, mem: list):
        """Append a memory to the memory system.

        Args
        ----
        mem: A memory in a quadruple format (i.e., (head, relation, tail, timestamp))

        """
        if self._frozen:
            error_msg = "The memory system is frozen!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        if len(self.entries) >= self.capacity:
            error_msg = "memory is full! can't add more."
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Adding a new memory entry {mem} ...")
        self.entries.append(mem)
        logging.info(f"memory entry {mem} added!")

    @property
    def is_empty(self) -> bool:
        """Return true if full."""
        return len(self.entries) == 0

    @property
    def is_full(self) -> bool:
        """Return true if full."""
        return len(self.entries) == self.capacity

    @property
    def is_frozen(self):
        """Is frozen?"""
        return self._frozen

    def freeze(self):
        """Freeze the memory so that nothing can be added / deleted."""
        self._frozen = True

    def pretrain_semantic(self, pretrain_path: str, creation_time: int) -> int:
        """Pretrain the semantic memory system from ConceptNet.

        Args
        ----
        pretrain_path: the path to the pretrained semantic memory.
        creation_time: the unix time, in seconds, where the agent is first created.

        Returns
        -------
        free_space: free space that can be added to the episodic memory system.

        """
        assert self.type == "semantic"
        logging.debug("Populating the semantic memory system with prior knowledge ...")
        pretrained = read_json(pretrain_path)
        for head, relation_head in pretrained.items():
            if self.is_full:
                break

            relation = list(relation_head.keys())[0]
            tail = relation_head[relation][0]
            # timestamp is the time when the semantic memory was first stored
            mem = [head, relation, tail, creation_time]
            logging.debug(f"adding {mem} to the semantic memory system ...")
            self.add(mem)

        self.freeze()
        logging.info("The semantic memory system is frozen!")
        free_space = self.capacity - len(self.entries)
        self.capacity = len(self.entries)
        logging.info(
            f"The semantic memory is pretrained and frozen. The remaining space "
            f"{free_space} will be returned. Now the capacity of the semantic memory "
            f"system is {self.capacity}"
        )

        return free_space

    def get_duplicate_heads(self, head: str, entries: list = None) -> list:
        """Find if there are duplicate heads and return its index.

        At the moment, this is simply done by matching string values. In the end, an
        RL agent has to learn this by itself.

        Args
        ----
        head: (e.g., Tae's laptop)

        Returns
        -------
        duplicates: a list of memories. Every memory in this list has the same head as that
            of the observation (i.e., (head, relation, tail, timestamp)). None is returned
            when no duplicate heads were found.

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

    def get_oldest_memory(self, entries: list = None) -> list:
        """Get the oldest memory in the system.

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

        mem = sorted(entries, key=lambda x: x[-1])[0]
        logging.info(f"{mem} is the oldest memory in the entries.")

        return mem

    def get_latest_memory(self, entries: list = None) -> list:
        """Get the latest memory in the system.

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
        logging.info(f"{mem} is the latest memory in the entries.")

        return mem

    def get_similar(self, entries: list = None):
        """Find N episodic memories that can be compressed into one semantic.

        At the moment, this is simply done by matching string values. In the end, an
        RL agent has to learn this by itself.

        Returns
        -------
        episodic_memories: similar episodic memories
        semantic_memory: encoded (compressed) semantic memory in a quadruple format
            (i.e., (head, relation, tail, timestamp))

        """
        logging.debug("looking for episodic entries that can be compressed ...")
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        assert self.type == "episodic"
        # -1 removes the timestamps from the quadruples
        semantic_possibles = [[e.split()[-1] for e in entry[:-1]] for entry in entries]
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
            # The timestamp of this semantic memory is the last observation time of the
            # latest episodic memory.
            semantic_memory.append(episodic_memories[-1][-1])
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

    def forget_FIFO(self) -> None:
        """Forget the oldest entry in the memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them. In the end, an RL agent has to learn this by itself.

        """
        logging.debug("forgetting the oldest memory (FIFO)...")

        mem = self.get_oldest_memory()
        self.forget(mem)

    def forget_random(self) -> None:
        """Forget a uniform-random memory in the memory system."""
        logging.warning("forgetting a random memory using a uniform distribution ...")
        mem = random.choice(self.entries)
        self.forget(mem)


def load_questions(path: str) -> dict:
    """Load premade questions.

    Args
    ----
    path: path to the question json file.

    """
    logging.debug(f"loading questions from {path}...")
    questions = read_json(path)
    logging.info(f"questions loaded from {path}!")

    return questions


def select_a_question(step: int, data: dict, questions: dict, split: str) -> list:
    """Select a question uniform-randomly.

    Args
    ----
    step: the step of the RL agent.
    data: train, val, test data splits in dict.
    questions: questions in train, val, and test splits
    split: one of train, val, or test.

    Returns
    -------
    question: a triple (i.e., (head, relation, tail))

    """
    logging.debug("selecting a question ...")
    assert split in ["train", "val", "test"]

    if step < len(data[split]) - 1:
        idx_q = step + 1
    else:
        idx_q = step

    question = random.sample(questions[split][idx_q], 1)[0]
    logging.info(f"question {question} was chosen!")

    return question


def answer_NRO(question: list, memory: Memory) -> int:
    """Answer the question (NRO: New replaces old).

    If object X was found at Y and then later on found Z, then the this strategy answers
    Z, instead of Y.

    Args
    ----
    question: a triple (i.e., (head, relation, tail))
    memory: Memory object

    Returns
    -------
    reward: -1 if no heads were found, +1 if the retrieved memory matches both head and
        tail, and 0 if the retrieved memory only matches the head, not the tail.

    """
    query_head = question[0]
    query_tail = question[2]
    duplicates = memory.get_duplicate_heads(query_head, memory.entries)
    if duplicates is None:
        logging.info("no relevant memories were found in the entries.")
        reward = -1
    else:
        logging.info(f"{len(duplicates)} relevant memories were found in the entries!")
        mem = memory.get_latest_memory(duplicates)
        tail = mem[2]
        if tail == query_tail:
            logging.info(f"NRO retrieved the relevant memory {mem}!")
            reward = 1
        else:
            logging.info("The object was there in the past, not now!")
            reward = 0

    return reward


def answer_random(question: list, memory: Memory) -> int:
    """Answer the question with a uniform-randomly chosen memory.

    Args
    ----
    question: a triple (i.e., (head, relation, tail))
    memory: Memory object

    Returns
    -------
    reward: -1 if no heads were found, +1 if the retrieved memory matches both head and
        tail, and 0 if the retrieved memory only matches the head, not the tail.

    """
    query_head = question[0]
    query_tail = question[2]

    mem = random.choice(memory.entries)
    head = mem[0]
    tail = mem[2]

    if head != query_head:
        logging.info("The retrieved memory is not correct for the question!")
        reward = -1
    else:
        if tail == query_tail:
            logging.info(f"retrieved the relevant memory {mem}!")
            reward = 1
        else:
            logging.info("The object was there in the past, not now!")
            reward = 0

    return reward


def ob2sem(ob: list):
    """Turn an episodic observation into a semantic memory.

    At the moment, this is simply done by removing the names from the head and the tail
    In the end, an RL agent has to learn this by itself.

    Args
    ----
    ob: An observation in a quadruple format
        (i.e., (head, relation, tail, timestamp))

    """
    logging.debug(f"Turning an observation {ob} into a semantic memory ...")
    # split to remove the name
    head = ob[0].split()[-1]
    relation = ob[1]
    tail = ob[2].split()[-1]
    if len(ob) == 4 and isinstance(ob[3], int):
        timestamp = ob[3]
    elif len(ob) == 3:
        logging.warning("this must be a question, not an observation!")
        timestamp = None
    else:
        error_msg = "something ain't right"
        logging.error(error_msg)
        raise ValueError(error_msg)

    sem_mem = [head, relation, tail, timestamp]
    logging.info(f"Observation {ob} is now a semantic memory {sem_mem}")

    return sem_mem
