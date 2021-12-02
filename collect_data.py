"""Collect data from ConceptNet."""
import logging
import os
import random
import time
from collections import Counter

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
        api_url: str
    ):
        """Data (conceptnet) collector class.

        Args
        ----
        relation: See https://github.com/commonsense/conceptnet5/wiki/Relations for
            all relations.
        conceptnet_data_path: Where to save raw queried conceptnet data path
        conceptnet_data_refresh: Whether to download the conceptnet data again or not.
        semantic_knowledge_path: Where to save pre-trained semantic (factual) knowledge
        weighting_mode: "highest" chooses the one with the highest weight, "weighted"
            chooses all of them by weight, and null chooses every single one of them
            without weighting.
        api_url: e.g., http://api.conceptnet.io/, http://127.0.0.1:8084/, etc.

        """
        self.relation = relation
        self.relation_simple = self.relation.split("/")[-1]

        self.conceptnet_data_path = conceptnet_data_path
        self.conceptnet_data_refresh = conceptnet_data_refresh
        self.semantic_knowledge_path = semantic_knowledge_path
        self.weighting_mode = weighting_mode
        self.api_url = api_url

        self.read_mscoco()
        self.read_names()
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

    def get_from_conceptnet(self) -> None:
        """Get data from ConceptNet API by HTTP get query."""
        logging.debug("retrieving data from conceptnet ...")

        if self.conceptnet_data_refresh:
            self.raw_data = {}
            for object_category in tqdm(self.mscoco):
                query = (
                    f"{self.api_url}"
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


def main(**kwargs) -> None:
    """Collect data. See ./collect_data.yaml for the config."""
    dc = DataCollector(**kwargs)
    dc.get_from_conceptnet()


if __name__ == "__main__":
    config = read_yaml("./collect_data.yaml")
    print("Arguments:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(**config)
