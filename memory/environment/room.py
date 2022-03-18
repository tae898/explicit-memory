import logging
import os
import random
from copy import deepcopy
from typing import List, Tuple

import gym

from ..memory import EpisodicMemory
from ..utils import read_json

CORRECT = 1
WRONG = 0

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RoomEnv(gym.Env):

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        semantic_knowledge_path: str = "./data/semantic-knowledge.json",
        names_path: str = "./data/top-human-names",
        weighting_mode: str = "weighted",
        probs: dict = {"commonsense": 0.5, "new_location": 0.5, "new_object": 0.5},
        renew_steps: int = 5,
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": False,
        },
        max_step: int = 1000,
        disjoint_entities: bool = True,
    ) -> None:
        """"""
        super().__init__()
        logging.debug("Creating an Observation-Question-Answer generator object ...")
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

        assert len(self.relations) == 1, "At the moment there is only one relation."

        self.names = self.read_names(
            names_path, self.limits["names"], self.limits["allow_spaces"]
        )

        assert len(self.names) <= len(self.heads)

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
        self.probs = probs
        self.renew_steps = renew_steps
        self.max_step = max_step

    def reset(self) -> None:
        """Reset the environment."""
        self.step_counter = 0
        random.shuffle(self.names)
        random.shuffle(self.heads)
        self.room = []

        for name, head in zip(self.names, self.heads):
            relation = self.relations[0]  # At the moment there is only one relation.
            tail = self.generate_tail(head, relation)
            self.room.append([f"{name}'s {head}", relation, f"{name}'s {tail}"])

        observed = deepcopy(random.choice(self.room))
        observed.append(self.step_counter)
        question = random.choice(self.room)
        self.answer = EpisodicMemory.remove_name(question[-1])
        question = question[:2]

        return (observed, question)

    def generate_tail(self, head: str, relation: str) -> str:
        """head shouldn't include a human name."""

        if random.random() < self.probs["commonsense"]:
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

        return tail

    def renew(self):
        """Renew the room.

        With the chance of probs["new_location"], an object will be placed at a new
        location.

        When the object is placed at a new locaiton, with the chance of
        probs["commonsense"], an object will be placed at a commonsense-knowledge spot.

        With the chance of probs["new_object"], the person with the object will have
        a new random object.

        """
        room = []
        for head, relation, tail in self.room:
            name1, head = EpisodicMemory.split_name_entity(head)
            name2, tail = EpisodicMemory.split_name_entity(tail)

            assert name1 == name2

            if random.random() < self.probs["new_object"]:
                while True:
                    new_head = random.choice(self.heads)
                    if new_head != head:
                        head = new_head
                        tail = self.generate_tail(head, relation)
                        break
            else:
                if random.random() < self.probs["new_location"]:
                    while True:
                        new_tail = self.generate_tail(head, relation)
                        if new_tail != tail:
                            tail = new_tail
                            break

            room.append(
                [
                    f"{name1}'s {deepcopy(head)}",
                    deepcopy(relation),
                    f"{name2}'s {deepcopy(tail)}",
                ],
            )
        self.room = deepcopy(room)

    def step(self, action: str):

        if str(action).lower() == self.answer.lower():
            logging.info(
                f"The prediction ({action}) matches the answer ({self.answer})!"
            )
            reward = CORRECT

        else:
            logging.info(
                f"The prediction ({action}) does NOT match the answer ({self.answer})!"
            )
            reward = WRONG

        self.step_counter += 1

        if self.step_counter % self.renew_steps == 0:
            logging.debug("renewing the room ...")
            self.renew()

        observed = deepcopy(random.choice(self.room))
        observed.append(self.step_counter)
        question = random.choice(self.room)

        self.answer = EpisodicMemory.remove_name(question[-1])
        question = question[:2]

        info = {}

        if self.step_counter >= self.max_step:
            done = True
        else:
            done = False

        return (observed, question), reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        else:
            pass

    def close(self):
        pass

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
