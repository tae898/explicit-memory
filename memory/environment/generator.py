import logging
import os
import random
from typing import List, Tuple
import math

from ..utils import read_json
from ..memory import EpisodicMemory

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
        time_start_at: int = 0,
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": False,
        },
        disjoint_entities: bool = True,
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
        time_start_at: beginning number of timestamp
        limits: Limit the heads, tails per head, and the number of names. For example,
            this can be {"heads": 10, "tails": 1, "names" 10, "allow_spaces": True}
        disjoint_entities: Whether to force that there are no common elements between
            entities.

        """
        logging.debug("Creating an Observation-Question-Answer generator object ...")
        self.max_history = max_history
        self.history = []
        self.limits = limits

        if set(list(self.limits.values())) != {None}:
            logging.warning(f"The obserations will be limited by {self.limits}")
        (
            self.semantic_knowledge,
            self.heads,
            self.relations,
            self.tails,
        ) = self.load_semantic_knowledge(
            semantic_knowledge_path,
            limit_heads=self.limits["heads"],
            limit_tails=self.limits["tails"],
            allow_spaces=self.limits["allow_spaces"],
            disjoint_entities=disjoint_entities,
        )

        self.names = self.read_names(
            names_path, self.limits["names"], self.limits["allow_spaces"]
        )
        if disjoint_entities:
            lhs = len(set(self.relations + self.names + self.heads + self.tails))
            rhs = (
                len(self.relations)
                + len(self.names)
                + len(self.heads)
                + len(self.tails)
            )

            assert lhs == rhs

        self.weighting_mode = weighting_mode
        self.commonsense_prob = commonsense_prob

        self.time_start_at = time_start_at
        self.reset_time()

        logging.info(
            "An Observation-Question-Answer generator object is generated. "
            f"timestamp is set to {time_start_at}."
        )

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

        return True

    def clear_history(self) -> None:
        """Remove all history."""
        logging.debug(f"Reseting the history of length {len(self.history)}...")
        self.history.clear()
        logging.info("Reseting the history is done!")

    def reset_time(self) -> None:
        """Reset time."""
        self.timestamp = self.time_start_at
        logging.info(f"time reset to {self.time_start_at}")

    def reset(self) -> None:
        """Reset the generator."""
        self.clear_history()
        self.reset_time()

    @property
    def is_full(self) -> bool:
        """Return true if full."""

        return len(self.history) == self.max_history

    @staticmethod
    def load_semantic_knowledge(
        path: str,
        limit_heads: int = None,
        limit_tails: int = None,
        allow_spaces: bool = False,
        disjoint_entities: bool = True,
    ) -> Tuple[list, list, list, list]:
        """Load saved semantic knowledge.

        Args
        ----
        path: the path to the pretrained semantic memory.
        limit_heads: Limit the number of heads (e.g., 10)
        limit_tails: Limit the number of tails per heads (e.g., 1)
        allow_spaces: Whether to include words that have spaces
            (e.g., corner of two streets)
        disjoint_entities: Whether to force that there are no common elements between
            entities.

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
        if disjoint_entities:
            logging.warning("Tails that are heads will be removed.")
            semantic_knowledge = {
                key: {
                    key_: [tail for tail in val_ if tail["tail"] not in heads]
                    for key_, val_ in val.items()
                }
                for key, val in semantic_knowledge.items()
            }

        semantic_knowledge = {
            key: {
                key_: val_
                for key_, val_ in val.items()
                if len([tail for tail in val_ if tail["tail"]]) > 0
            }
            for key, val in semantic_knowledge.items()
        }
        semantic_knowledge = {
            key: val for key, val in semantic_knowledge.items() if len(val) > 0
        }
        logging.info("empty entities are removed")

        # sort the semantic knowledge by its highest weight to be sure.
        semantic_knowledge = {
            key: {
                key_: sorted(val_, key=lambda x: -x["weight"])
                for key_, val_ in val.items()
            }
            for key, val in semantic_knowledge.items()
        }

        if not allow_spaces:
            semantic_knowledge = {
                key: {
                    key_: [v for v in val_ if len(v["tail"].split("_")) == 1]
                    for key_, val_ in val.items()
                }
                for key, val in semantic_knowledge.items()
                if (len(key.split("_"))) == 1
            }

        if limit_heads:
            logging.warning(f"Limiting the number of heads to {limit_heads} ...")
            semantic_knowledge = {
                key: val
                for idx, (key, val) in enumerate(semantic_knowledge.items())
                if idx < limit_heads
            }

        if limit_tails:
            logging.warning(
                f"Limiting the number of tails per head to {limit_tails} ..."
            )
            semantic_knowledge = {
                key: {key_: val_[:limit_tails] for key_, val_ in val.items()}
                for key, val in semantic_knowledge.items()
            }

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
    def read_names(
        path: str = "./data/top-human-names",
        limit_names: int = None,
        allow_spaces: bool = False,
    ) -> list:
        """Read 20 most common names.

        Args
        ----
        path: The path to the top 20 human name list.
        limit_names: Limit the number of names
        allow_spaces: Whether to include words that have spaces
            (e.g., corner of two streets)

        Returns
        -------
        names: human names (e.g., James)

        """
        logging.debug(f"Reading {path} ...")
        with open(path, "r") as stream:
            names = stream.readlines()
        names = [line.strip() for line in names]

        if not allow_spaces:
            names = [name for name in names if len(name.split("_")) == 1]

        if limit_names:
            logging.warning(f"The number of names will be limited to {limit_names}")
            names = sorted(names, key=len)
            names = names[:limit_names]

        logging.info(f"Reading {path} complete! There are {len(names)} names in total")

        return names

    def generate_observation(self) -> list:
        """Generate an observation.

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

        ob = [head, relation, tail, self.timestamp]
        logging.info(f"A new observation generated: {ob}")

        self.timestamp += 1

        return ob

    def generate_question_answer(self, recent_more_likely: bool = True) -> List[str]:
        """Generate a question based on the observation history.

        The recent observations are more likely to be questions.

        Args
        ----
        recent_more_likely: put more weights on the recent observations when random
            sampling so that they are more likely

        Returns
        -------
        question_answer: [head, relation, tail]. `head` and `relation` together make
             a question and `tail` is the answer.

        """
        assert len(self.history) != 0
        logging.debug("Generating a question and answer based on the history ...")
        heads_covered = []
        questions = []
        for ob in self.history[::-1].copy():  # we start from latest first
            head = ob[0]
            # This way the question only asks the latest location of an object
            if head not in heads_covered:
                questions.append(ob)
                heads_covered.append(head)

        # now we start from oldest first
        questions = questions[::-1]
        if recent_more_likely:
            # The recent observations are more likely to be questions
            question_answer = random.choices(
                # questions, weights=[math.exp(i + 1) for i in range(len(questions))], k=1
                questions, weights=[(i + 1) for i in range(len(questions))], k=1

            )[0]
        else:
            question_answer = random.choice(questions)
        # :-1 removes the timestamp
        question_answer = question_answer[:-1]
        question_answer[-1] = EpisodicMemory.remove_name(question_answer[-1])
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

    def generate_with_history(
        self,
        generate_qa: bool = True,
        recent_more_likely: bool = True,
    ) -> Tuple[list, List[str]]:
        """Generate an observation, question, and answer and add in the history.

        Everything comes from a uniform random distribution.

        Args
        ----
        generate_qa: Whether to generate a question and answer along with
            observation or not.
        recent_more_likely: put more weights on the recent observations when random
            sampling so that they are more likely

        Returns
        --------
        ob: observation `[head, relation, tail, timestamp]`
        qa: [head, relation, tail]. `head` and `relation` together make
             a question and `tail` is the answer

        """
        ob = self.generate_observation()

        if self.is_full:
            logging.info(
                "history is full, the oldest observation will be removed to add a new one."
            )
            self.history.pop(0)

        self.add_observation_to_history(ob)
        logging.info("The new observation is added to the history.")
        if generate_qa:
            qa = self.generate_question_answer(recent_more_likely)
        else:
            qa = None

        return ob, qa

    def is_possible_observation(self, ob: list) -> bool:
        """Test if a given observation can be generated with the current generator.

        Args
        ----
        ob: observation in [head, relation, tail, timestamp]

        Returns
        -------
        True or False

        """
        if not isinstance(ob, list):
            return False
        if len(ob) != 4:
            return False

        head = ob[0]
        name1, obj = EpisodicMemory.split_name_entity(head)
        relation = ob[1]
        tail = ob[2]
        name2, location = EpisodicMemory.split_name_entity(tail)
        timestamp = ob[3]

        if name1 != name2:
            return False
        if obj not in self.heads:
            return False
        if relation not in self.relations:
            return False
        if location not in self.tails:
            return False
        if timestamp < 0:
            return False

        return True

    def is_possible_qa(self, qa_epi: list) -> bool:
        """Test if it is a possible question.

        Args
        ----
        qa_epi: [head, relation, tail]. `head` and `relation` together make
             a question and `tail` is the answer. The answer does not contain the name.

        Returns
        -------
        True or False

        """
        if not isinstance(qa_epi, list):
            return False
        if len(qa_epi) != 3:
            return False

        head = qa_epi[0]
        name1, head = EpisodicMemory.split_name_entity(head)
        relation = qa_epi[1]
        tail = qa_epi[2]

        if head not in self.heads:
            return False
        if relation not in self.relations:
            return False
        if tail not in self.tails:
            return False

        return True
