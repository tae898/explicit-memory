import logging
import os
from pprint import pformat

from gym.spaces import Space

from .generator import OQAGenerator

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MemorySpace(Space):
    def __init__(
        self,
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
    ) -> None:
        """

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
        time_start_at: beginning number of timestamp
        limits: Limit the heads, tails per head, and the number of names. For example,
            this can be {"heads": 10, "tails": 1, "names" 10, "allow_spaces": True}
        disjoint_entities: Whether to force that there are no common elements between
            entities.
        """
        self.oqag_dummy = OQAGenerator(
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            time_start_at=time_start_at,
            limits=limits,
            disjoint_entities=disjoint_entities,
        )
        super().__init__()

    def sample(self):
        """Sample a state."""
        self.oqag_dummy.reset()

        ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
        question = qa_epi[:2]

        state = {"observation": ob, "question": question}

        return state

    def contains(self, x):
        if len(x) != 2:
            return False

        ob, qa_epi = x[0], x[1]

        if not isinstance(ob, list):
            return False

        if len(ob) != 4:
            return False

        if not isinstance(qa_epi, list):
            return False

        if len(qa_epi) != 3:
            return False

        if not self.oqag_dummy.is_possible_observation(ob):
            return False

        if not self.oqag_dummy.is_possible_qa(qa_epi):
            return False

        return True

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

    def __eq__(self, other):
        if self.oqag_dummy.max_history != other.oqag_dummy.max_history:
            return False
        if self.oqag_dummy.heads != other.oqag_dummy.heads:
            return False
        if self.oqag_dummy.relations != other.oqag_dummy.relations:
            return False
        if self.oqag_dummy.tails != other.oqag_dummy.tails:
            return False
        if self.oqag_dummy.names != other.oqag_dummy.names:
            return False
        if self.oqag_dummy.weighting_mode != other.oqag_dummy.weighting_mode:
            return False
        if self.oqag_dummy.commonsense_prob != other.oqag_dummy.commonsense_prob:
            return False

        return True
