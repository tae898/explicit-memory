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
        max_step: int = 1000,
        generator_params: dict = {
            "max_history": 128,
            "semantic_knowledge_path": "./data/semantic-knowledge.json",
            "names_path": "./data/top-human-names",
            "weighting_mode": "weighted",
            "commonsense_prob": 0.5,
            "time_start_at": 0,
            "limits": {
                "heads": None,
                "tails": None,
                "names": None,
                "allow_spaces": False,
            },
            "disjoint_entities": True,
        },
    ) -> None:
        """

        Args
        ----
        max_step: max step
        generator_params: generator_params
        """
        super().__init__()
        self.max_step = max_step
        self.generator_params = generator_params
        self.oqag = OQAGenerator(**generator_params)

        logging.info("MemoryEnv is successfully instantiated")

    def reset(self):
        """Reset the environment.

        This resets the data generator (i.e., OQAGenerator) and the step counter

        """
        self.oqag.reset()
        self.step_counter = 0

        logging.info("MemoryEnv has been successfully reset")

    def step(self, mode: str, action: str = None):

        if mode == "observation":
            ob, _ = self.oqag.generate_with_history(generate_qa=False)
            self.answer = None

            state = ob
            reward = None
            done = False
            info = {}

        elif mode == "question":
            qa = self.oqag.generate_question_answer()
            question = qa[:2]
            self.answer = qa[2]

            state = question
            reward = None
            done = False
            info = {}

        elif mode == "answer":
            assert self.answer is not None

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

            if self.step_counter >= self.max_step:
                done = True
            else:
                done = False

            state = None
            info = {}

        else:
            raise ValueError

        return state, reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        else:
            pass

    def close(self):
        pass
