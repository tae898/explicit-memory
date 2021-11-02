import os
import logging
from pprint import pformat


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

    def forget(self, mem: list = None):
        """forget the given memory. if the memory to be forgotten is not specified, then
        forget the oldest memory.

        Args
        ----
        mem: A memory (i.e., (head, relation, tail, timestamp))
        """
        assert not self._frozen
        if mem is None:
            logging.debug(f"finding the oldest memory ...")
            mem = sorted(self.entries, key=lambda x: x[-1])[0]

        logging.debug(f"Forgetting {mem} ...")
        self.entries.remove(mem)
        logging.info(f"{mem} forgotten!")

    def add(self, mem: list):
        """Append elem to the memory

        Args
        ----
        mem: A memory (i.e., (head, relation, tail, timestamp))
        """

        assert not self._frozen
        if len(self.entries) >= self.capacity:
            error_msg = "memory is full! can't add more."
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Adding a new memory entry {mem} ...")
        self.entries.append(mem)
        logging.info(f"memory entry {mem} added!")

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
