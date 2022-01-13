import logging
import os

import gym

from memory.constants import CORRECT, WRONG

from .generator import OQAGenerator

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MemoryEnv(gym.Env):
    """Episodic and semantic memory environment compatiable with the gym interface.

    This is essentially a POMDP, since the agent is not observing the entire history.
    This enforces the agent to have memory strategies.

    """

    metadata = {"render.modes": ["console"]}

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
        super().__init__()

        self.oqag = OQAGenerator(
            max_history=max_history,
            semantic_knowledge_path=semantic_knowledge_path,
            names_path=names_path,
            weighting_mode=weighting_mode,
            commonsense_prob=commonsense_prob,
            time_start_at=time_start_at,
            limits=limits,
            disjoint_entities=disjoint_entities,
        )

        self.step_counter = 0
        self.max_history = max_history

        logging.info("MemoryEnv is successfully instantiated")

    def reset(self):
        """Reset the environment.
        
        This resets the data generator (i.e., OQAGenerator) and the step counter

        """
        self.oqag.reset()
        self.step_counter = 0

        ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
        question = qa_epi[:2]
        self.answer = qa_epi[2]

        state = {"observation": ob, "question": question}

        logging.info("MemoryEnv has been successfully reset")

        return state

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

        ob, qa_epi = self.oqag.generate_with_history(generate_qa=True)
        question = qa_epi[:2]
        self.answer = qa_epi[2]

        state = {"observation": ob, "question": question}

        self.step_counter += 1

        if self.step_counter >= self.max_history:
            done = True
        else:
            done = False

        info = {}

        return state, reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        else:
            print(self.M_e.entries)

    def close(self):
        pass
