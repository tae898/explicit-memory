import random
import os
import logging
import time
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

        self.load_semantic_knowledge(semantic_knowledge_path)
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

        self.read_names(names_path)
        self.weighting_mode = weighting_mode
        self.commonsense_prob = commonsense_prob
        logging.info("An Observation-Question-Answer generator object is generated!")

    def reset(self) -> None:
        """Reset the generator."""
        logging.debug(f"Reseting the history of length {len(self.history)}...")
        self.history.clear()
        logging.info("Reseting the history is done!")

    @property
    def is_full(self) -> bool:
        """Return true if full."""

        return len(self.history) == self.max_history

    def load_semantic_knowledge(self, path: str) -> None:
        """Load saved semantic knowledge.

        Args
        ----
        path: the path to the pretrained semantic memory.

        """
        logging.debug(f"loading the semantic knowledge from {path}...")
        self.semantic_knowledge = read_json(path)
        logging.info(f"semantic knowledge successfully loaded from {path}!")

    def read_names(self, path: str = "./data/top-human-names") -> None:
        """Read 20 most common names.

        Args
        ----
        path: The path to the top 20 human name list.

        """
        logging.debug(f"Reading {path} ...")
        with open(path, "r") as stream:
            self.names = stream.readlines()
        self.names = [line.strip() for line in self.names]
        logging.info(
            f"Reading {path} complete! There are {len(self.names)} names in total"
        )

    def generate(self):
        """Generate an observation, question, and answer.

        Everything comes from a uniform random distribution.

        Args
        ----
        commonsense_prob: the probability of an observation being covered by a
            commonsense

        """
        if self.is_full:
            raise ValueError(f"History {len(self.history)} is full")

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

        # unix timestamp in seconds
        timestamp = time.time()
        ob = [head, relation, tail, timestamp]
        logging.info(f"A new observation generated: {ob}")

        self.history.append(ob)
        logging.info("The new observation is added to the history.")
        question_answer = self.generate_question()

        return ob, question_answer

    def generate_question(self):
        """Generate a question based on the observation history."""
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
