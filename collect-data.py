"""Collect data from ConceptNet."""
import logging
import os
import random
from collections import Counter
import time

import requests
from tqdm import tqdm

from memory.utils import read_json, read_yaml, write_csv, write_json

# for reproducibility
random.seed(42)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DataCollector:
    """Data (conceptnet) collector class."""

    def __init__(
        self,
        relation: str,
        conceptnet_data_path: str,
        conceptnet_data_refresh: bool,
        semantic_knowledge_path: str,
        weighting_mode: bool,
        # num_repeat: int = None,
        # dataset_stats_path: str = None,
        # common_obs_path: str = None,
        # episodic_factor: int = None,
        # noncommon_obs_path: str = None,
        # all_obs_path: str = None,
        # final_data_path: str = None,
        # val_ratio: float = None,
        # test_ratio: float = None,
        # question_path: str = None,
        # delay_seconds: int = None,
        # time_zero: bool = None,
    ):
        """Data (conceptnet) collector class.

        Args
        ----
        relation: See https://github.com/commonsense/conceptnet5/wiki/Relations for
            all relations.
        conceptnet_data_path: Where to save raw queried conceptnet data path
        conceptnet_data_refresh: Whether to download the conceptnet data again or not.
        dataset_stats_path: Where to save the stats of the raw data
        semantic_knowledge_path: Where to save pre-trained semantic (factual) knowledge
        common_obs_path: Where to save common observations
        weighting_mode: "highest" chooses the one with the highest weight, "weighted"
            chooses all of them by weight, and null chooses every single one of them
            without weighting.
        num_repeat: Number of repeats for common observations to increase data size
        episodic_factor: factor for noncommon observations. 2 means it will generate
            twice as much
        noncommon_obs_path: Where to save noncommon observations
        all_obs_path: Where to save all observations
        final_data_path: Where to save train, val, test splits
        val_ratio: validation split ratio
        test_ratio: test split ratio
        question_path: where to save questions
        delay_seconds: mock time delay between observations. 3600 seconds is one hour
        time_zero: set the earliest time to zero for convenience

        """
        self.relation = relation
        self.relation_simple = self.relation.split("/")[-1]

        self.conceptnet_data_path = conceptnet_data_path
        self.conceptnet_data_refresh = conceptnet_data_refresh
        self.semantic_knowledge_path = semantic_knowledge_path
        self.weighting_mode = weighting_mode

        # self.num_repeat = num_repeat
        # self.dataset_stats_path = dataset_stats_path
        # self.common_obs_path = common_obs_path
        # self.episodic_factor = episodic_factor
        # self.noncommon_obs_path = noncommon_obs_path
        # self.all_obs_path = all_obs_path
        # self.final_data_path = final_data_path
        # self.val_ratio = val_ratio
        # self.test_ratio = test_ratio
        # self.question_path = question_path
        # self.delay_seconds = delay_seconds
        # self.time_zero = time_zero

        self.read_mscoco()
        self.read_names()
        self.read_dirty_tails()
        os.makedirs("./data", exist_ok=True)

        logging.info("DataCollector object successfully instantiated!")

    def read_mscoco(self, path: str = "./data/ms-coco-80-categories") -> None:
        """Return ms coco 80 object categories.

        Args
        ----
        path: The path to the mscoco object category list.

        """
        logging.debug(f"Reading {path} ...")
        with open(path, "r") as stream:
            self.mscoco = stream.readlines()
        self.mscoco = [line.strip() for line in self.mscoco]
        self.mscoco = ["_".join(foo.split()) for foo in self.mscoco]
        logging.info(
            f"Reading {path} complete! There are {len(self.mscoco)} object categories "
            "in total."
        )

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

    def read_dirty_tails(self, path: str = "./data/dirty_tails") -> None:
        """Read dirty tails.

        Args
        ----
        path: The path to the dirty tail list.

        """
        logging.debug(f"Reading {path} ...")
        with open(path, "r") as stream:
            self.dirty_tails = stream.readlines()
        self.dirty_tails = [line.strip() for line in self.dirty_tails]
        logging.info(
            f"Reading {path} complete! There are {len(self.dirty_tails)} dirty tails!"
        )

    def get_from_conceptnet(self) -> None:
        """Get data from ConceptNet API by HTTP get query."""
        logging.debug("retrieving data from conceptnet ...")

        if self.conceptnet_data_refresh:
            self.raw_data = {}
            for object_category in tqdm(self.mscoco):
                query = (
                    "http://api.conceptnet.io/"
                    f"query?start=/c/en/{object_category}&rel={self.relation}"
                )
                logging.debug(f"making an HTTP get request with query {query}")
                response = requests.get(query).json()

                logging.info(
                    f"{len(response['edges'])} tails (entities) found for "
                    f"{object_category}!"
                )
                if len(response["edges"]) == 0:
                    continue

                self.raw_data[object_category] = []

                for edge in tqdm(response["edges"]):

                    if self.is_clean(edge):
                        self.raw_data[object_category].append(
                            {
                                "start": edge["start"],
                                "end": edge["end"],
                                "weight": edge["weight"],
                                "surfaceText": edge["surfaceText"],
                            }
                        )

            write_json(self.raw_data, self.conceptnet_data_path)
            logging.info(
                f"conceptconceptnet_data_path data retrieval done and saved at "
                f"{self.conceptnet_data_path}"
            )
        else:
            logging.debug(
                f"Loading the existing conceptnet data from "
                f"{self.conceptnet_data_path}..."
            )
            self.raw_data = read_json(self.conceptnet_data_path)
            logging.info(
                f"Conceptnet data successfully loaded from {self.conceptnet_data_path}"
            )

        logging.debug(
            f"Creating semantic knowledge at {self.semantic_knowledge_path} ..."
        )
        self.semantic_knowledge = {}

        for key, val in self.raw_data.items():
            head = key

            self.semantic_knowledge[head] = {self.relation_simple: []}

            if self.weighting_mode == "highest" and len(val) > 0:
                tail = sorted(val, key=lambda x: x["weight"], reverse=True)[0]
                tail = tail["end"]["@id"].split("/")[-1]
                self.semantic_knowledge[head][self.relation_simple].append(
                    {"tail": tail, "weight": 1}
                )

            else:
                for val_ in val:
                    tail = val_["end"]["@id"].split("/")[-1]

                    weight = 1 if self.weighting_mode is None else round(val_["weight"])

                    self.semantic_knowledge[head][self.relation_simple].append(
                        {"tail": tail, "weight": weight}
                    )

        write_json(self.semantic_knowledge, self.semantic_knowledge_path)
        logging.info(f"semantic knowledge saved at {self.semantic_knowledge_path} ...")

    def is_clean(self, edge: dict) -> None:
        """See if conceptnet query is clean or not.

        I really tried to accept as much noise as possible, since I didn't want
        to manually clean the data, but some are really dirty. I gotta clean them.

        Args
        ----
        edge: edge from the ConceptNet query output.

        """
        logging.debug(f"Checking if {edge} is clean or not ...")
        if edge["end"]["@id"].split("/")[-1] in self.dirty_tails:
            return False
        else:
            return True

    # def create_common_observations(self) -> None:
    #     """Create common sense observations using 20 human names.

    #     10 male and 10 female names were used from here
    #     https://www.ssa.gov/oact/babynames/decades/century.html
    #     """
    #     logging.debug("Creating dummy common observations ...")
    #     self.data_weighted = []
    #     self.semantic_knowledge = {}

    #     for key, val in self.raw_data.items():
    #         head = key

    #         self.semantic_knowledge[head] = {self.relation_simple: []}

    #         if self.weighting_mode == "highest" and len(val) > 0:
    #             tail = sorted(val, key=lambda x: x["weight"], reverse=True)[0]
    #             tail = tail["end"]["@id"].split("/")[-1]
    #             self.data_weighted.append((head, self.relation_simple, tail))
    #             self.semantic_knowledge[head][self.relation_simple].append(tail)

    #         else:
    #             for val_ in val:
    #                 tail = val_["end"]["@id"].split("/")[-1]
    #                 weight = round(val_["weight"])

    #                 if self.weighting_mode is None:
    #                     weight = 1
    #                 for _ in range(weight):
    #                     self.data_weighted.append((head, self.relation_simple, tail))

    #                 self.semantic_knowledge[head][self.relation_simple].append(tail)

    #     self.data_weighted = sorted(self.data_weighted)
    #     self.data_weighted = self.data_weighted * self.num_repeat

    #     random.shuffle(self.data_weighted)

    #     # Note that the observations are all commonsense.
    #     self.obs_semantic = []
    #     for head, edge, tail in self.data_weighted:
    #         rand_name = random.choice(self.names)
    #         head = rand_name + "'s " + head
    #         tail = rand_name + "'s " + tail
    #         self.obs_semantic.append((head, edge, tail))

    #     write_json(self.semantic_knowledge, self.semantic_knowledge_path)
    #     logging.info(f"semantic knowledge saved at {self.semantic_knowledge_path} ...")

    # def create_non_common_observations(self) -> None:
    #     """Create noncommon observations based on common observations."""
    #     logging.debug("Creating dummy noncommon observations ...")

    #     possible_locations = list(
    #         set(
    #             [
    #                 loc
    #                 for key, val in self.semantic_knowledge.items()
    #                 for loc in val[self.relation_simple]
    #             ]
    #         )
    #     )
    #     possible_locations = sorted(possible_locations)
    #     self.obs_episodic = []

    #     for obs_s in self.obs_semantic:
    #         head_s = obs_s[0].split()[-1]
    #         assert obs_s[1] == self.relation_simple

    #         name_head = obs_s[0].split("'")[0]
    #         name_tail = obs_s[2].split("'")[0]

    #         assert name_head == name_tail

    #         for _ in range(self.episodic_factor):
    #             while True:
    #                 location_random = random.choice(possible_locations)
    #                 if (
    #                     location_random
    #                     not in self.semantic_knowledge[head_s][self.relation_simple]
    #                 ):
    #                     break
    #             self.obs_episodic.append(
    #                 (
    #                     obs_s[0],
    #                     self.relation_simple,
    #                     name_tail + "'s " + location_random,
    #                 )
    #             )

    # def create_splits(self) -> None:
    #     """Create train, val, and test splits"""
    #     logging.debug("Creating train, val, and test splits ...")
    #     self.obs_all = self.obs_episodic + self.obs_semantic
    #     random.shuffle(self.obs_all)

    #     logging.debug("adding timestamps to the observations ...")
    #     current_time = int(time.time())
    #     for idx in range(len(self.obs_all)):
    #         current_time -= self.delay_seconds
    #         self.obs_all[idx] += (current_time,)

    #     if self.time_zero:
    #         logging.debug("zeroing the time ...")
    #         min_time = min([ob[-1] for ob in self.obs_all])

    #         for i in range(len(self.obs_all)):
    #             ob = list(self.obs_all[i])
    #             ob[-1] -= min_time
    #             ob = tuple(ob)
    #             self.obs_all[i] = ob
    #         logging.info("zeroing the time is complete!")

    #     self.obs_all = self.obs_all[::-1]
    #     logging.info("timestamps added to the observations!")

    #     write_csv(self.obs_semantic, self.common_obs_path)
    #     logging.info(f"dummy common observations saved at {self.common_obs_path}")

    #     write_csv(self.obs_episodic, self.noncommon_obs_path)
    #     logging.info(f"dummy noncommon observations saved at {self.common_obs_path}")

    #     write_csv(self.obs_all, self.all_obs_path)
    #     self.data_final = {}
    #     train_until = int(len(self.obs_all) * (1 - (self.val_ratio + self.test_ratio)))
    #     val_until = int(len(self.obs_all) * (1 - self.test_ratio))

    #     self.data_final["train"] = self.obs_all[:train_until]
    #     self.data_final["val"] = self.obs_all[train_until:val_until]
    #     self.data_final["test"] = self.obs_all[val_until:]

    #     assert (
    #         len(self.data_final["train"])
    #         + len(self.data_final["val"])
    #         + len(self.data_final["test"])
    #     ) == len(self.obs_all)

    #     write_json(self.data_final, self.final_data_path)
    #     logging.info("Splitting train, val, and test splits is done!")

    # def make_questions(self) -> None:
    #     """Create questions.

    #     The questions are designed that it only asks the latest location of an object.
    #     For example, let's say X was found at Y yesterday, and X was found at Z today.
    #     Then tomorrow when the question "Where is X?" is asked tomorrow, the agent
    #     has to say Z to get a reward.

    #     """
    #     self.questions_all = {"train": [], "val": [], "test": []}
    #     for SPLIT in tqdm(["train", "val", "test"]):
    #         logging.debug(f"creating questions for the {SPLIT} split ...")
    #         data = self.data_final[SPLIT]

    #         for idx in tqdm(range(len(data))):
    #             obs = data[:idx]
    #             obs = obs[::-1]

    #             questions = []
    #             heads_covered = []
    #             for ob in obs:
    #                 head = ob[0]
    #                 if head not in heads_covered:
    #                     questions.append(ob[:-1])
    #                     heads_covered.append(head)
    #             questions = questions[::-1]
    #             self.questions_all[SPLIT].append(questions)

    #         logging.info(f"questions for the {SPLIT} split are created!")

    #     write_json(self.questions_all, self.question_path)
    #     logging.info(f"questions saved at {self.question_path}")

    # def compute_dataset_stats(self) -> None:
    #     """Get basic data statistics."""
    #     logging.info("Computing dataset stats ...")
    #     raw_data_stats = {}

    #     num_train_samples = len(self.data_final["train"])
    #     num_val_samples = len(self.data_final["val"])
    #     num_test_samples = len(self.data_final["test"])

    #     raw_data_stats["num_train_samples"] = num_train_samples
    #     raw_data_stats["num_val_samples"] = num_val_samples
    #     raw_data_stats["num_test_samples"] = num_test_samples

    #     ############################################################################
    #     episodic_object_counts = dict(Counter([ob[0] for ob in self.obs_all]))
    #     episodic_object_counts = {
    #         key: val
    #         for key, val in sorted(
    #             episodic_object_counts.items(), key=lambda x: x[1], reverse=True
    #         )
    #     }

    #     num_episodic_objects = sum([val for key, val in episodic_object_counts.items()])
    #     num_unique_episodic_objects = len(episodic_object_counts)

    #     raw_data_stats["episodic_object_counts"] = episodic_object_counts
    #     raw_data_stats["num_episodic_objects"] = num_episodic_objects
    #     raw_data_stats["num_unique_episodic_objects"] = num_unique_episodic_objects

    #     ############################################################################

    #     ############################################################################
    #     semantic_object_counts = dict(
    #         Counter([ob[0].split()[-1] for ob in self.obs_all])
    #     )
    #     semantic_object_counts = {
    #         key: val
    #         for key, val in sorted(
    #             semantic_object_counts.items(), key=lambda x: x[1], reverse=True
    #         )
    #     }

    #     num_semantic_objects = sum([val for key, val in semantic_object_counts.items()])
    #     num_unique_semantic_objects = len(semantic_object_counts)

    #     raw_data_stats["semantic_object_counts"] = semantic_object_counts
    #     raw_data_stats["num_semantic_objects"] = num_semantic_objects
    #     raw_data_stats["num_unique_semantic_objects"] = num_unique_semantic_objects

    #     ############################################################################

    #     ############################################################################
    #     episodic_location_counts = dict(Counter([ob[2] for ob in self.obs_all]))
    #     episodic_location_counts = {
    #         key: val
    #         for key, val in sorted(
    #             episodic_location_counts.items(), key=lambda x: x[1], reverse=True
    #         )
    #     }

    #     num_episodic_locations = sum(
    #         [val for key, val in episodic_location_counts.items()]
    #     )
    #     num_unique_episodic_locations = len(episodic_location_counts)

    #     raw_data_stats["episodic_location_counts"] = episodic_location_counts
    #     raw_data_stats["num_episodic_locations"] = num_episodic_locations
    #     raw_data_stats["num_unique_episodic_locations"] = num_unique_episodic_locations

    #     ############################################################################

    #     ############################################################################
    #     semantic_location_counts = dict(
    #         Counter([ob[2].split()[-1] for ob in self.obs_all])
    #     )
    #     semantic_location_counts = {
    #         key: val
    #         for key, val in sorted(
    #             semantic_location_counts.items(), key=lambda x: x[1], reverse=True
    #         )
    #     }

    #     num_semantic_locations = sum(
    #         [val for key, val in semantic_location_counts.items()]
    #     )
    #     num_unique_semantic_locations = len(semantic_location_counts)

    #     raw_data_stats["semantic_location_counts"] = semantic_location_counts
    #     raw_data_stats["num_semantic_locations"] = num_semantic_locations
    #     raw_data_stats["num_unique_semantic_locations"] = num_unique_semantic_locations
    #     ############################################################################

    #     write_json(raw_data_stats, self.dataset_stats_path)
    #     logging.info(f"Dataset stats saved at {self.dataset_stats_path}")


def main(**kwargs) -> None:
    """Collect data. See ./collect-data.yaml for the config."""
    dc = DataCollector(**kwargs)

    dc.get_from_conceptnet()
    # dc.create_common_observations()
    # dc.create_non_common_observations()
    # dc.create_splits()
    # dc.compute_dataset_stats()
    # dc.make_questions()


if __name__ == "__main__":
    config = read_yaml("./collect-data.yaml")
    print("Arguments:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(**config)
