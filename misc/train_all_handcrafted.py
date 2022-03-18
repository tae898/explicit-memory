import argparse
import logging
import os
import time
import uuid
from copy import deepcopy
from datetime import datetime
from subprocess import Popen

from tqdm import tqdm

from memory.utils import read_yaml, write_json, write_yaml

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


config_base = {
    "seed": None,
    "training_params": {
        "device": "cuda",
        "precision": 32,
        "num_processes": 16,
        "gamma": 0.99,
        "learning_rate": 0.0001,
        "batch_size": 1,
        "callbacks": None,
    },
    "strategies": {
        "episodic_memory_manage": "oldest",
        "episodic_question_answer": "latest",
        "semantic_memory_manage": "weakest",
        "semantic_question_answer": "strongest",
        "episodic_to_semantic": "generalize",
        "episodic_semantic_question_answer": "episem",
        "pretrain_semantic": True,
        "capacity": {"episodic": 0, "semantic": 16},
        "policy_params": {"function_type": "mlp"},
    },
    "generator_params": {
        "max_history": 1024,
        "semantic_knowledge_path": "./data/semantic-knowledge.json",
        "names_path": "./data/top-human-names",
        "weighting_mode": "highest",
        "commonsense_prob": 0.5,
        "time_start_at": 0,
        "limits": {"heads": 40, "tails": 1, "names": 20, "allow_spaces": False},
        "disjoint_entities": True,
    },
}
seeds = [0, 1, 2, 3, 4]
capacities = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def generate_all_configs():
    logging.debug("Generating configs ...")
    configs = []

    # episodic only
    for capacity in capacities:
        for episodic_memory_manage in ["oldest", "random"]:
            for episodic_question_answer in ["latest", "random"]:
                for seed in seeds:
                    config = deepcopy(config_base)

                    config["strategies"]["capacity"]["episodic"] = capacity * 2
                    config["strategies"]["capacity"]["semantic"] = 0
                    config["strategies"][
                        "episodic_memory_manage"
                    ] = episodic_memory_manage
                    config["strategies"][
                        "episodic_question_answer"
                    ] = episodic_question_answer
                    config["seed"] = seed

                    configs.append(config)

    # semantic only
    for capacity in capacities:
        for semantic_memory_manage in ["weakest", "random"]:
            for semantic_question_answer in ["strongest", "random"]:
                for pretrain_semantic in [True, False]:
                    for seed in seeds:
                        config = deepcopy(config_base)

                        config["strategies"]["capacity"]["semantic"] = capacity * 2
                        config["strategies"]["capacity"]["episodic"] = 0
                        config["strategies"][
                            "semantic_memory_manage"
                        ] = semantic_memory_manage
                        config["strategies"][
                            "semantic_question_answer"
                        ] = semantic_question_answer
                        config["strategies"]["pretrain_semantic"] = pretrain_semantic
                        config["seed"] = seed

                        configs.append(config)

    # both episodic and semantic
    for capacity in capacities:
        for episodic_to_semantic in ["generalize", "noop"]:
            for episodic_semantic_question_answer in [
                "episem",
                "random",
            ]:
                for pretrain_semantic in [True, False]:
                    for seed in seeds:
                        config = deepcopy(config_base)

                        config["strategies"]["capacity"]["episodic"] = capacity
                        config["strategies"]["capacity"]["semantic"] = capacity

                        config["strategies"][
                            "episodic_to_semantic"
                        ] = episodic_to_semantic
                        config["strategies"][
                            "episodic_semantic_question_answer"
                        ] = episodic_semantic_question_answer
                        config["strategies"]["pretrain_semantic"] = pretrain_semantic

                        config["seed"] = seed

                        configs.append(config)

    logging.info(f"In total of {len(configs)} configs generated!")
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train ALL policies")
    parser.add_argument(
        "--num-procs", type=int, default=1, help="number of parallel processes to run"
    )

    num_procs = parser.parse_args().num_procs
    configs = generate_all_configs()

    if len(configs) % num_procs == 0:
        num_batches = len(configs) // num_procs
    else:
        num_batches = (len(configs) // num_procs) + 1

    logging.info(
        f"Training will begin on {len(configs)} configs with {num_batches} "
        f"batches using {num_procs} processes."
    )

    for i in tqdm(range(num_batches)):
        logging.debug(f"Running on {i+1} / {num_batches} batch ...")
        configs_batch = configs[i * num_procs : (i + 1) * num_procs]
        configs_batch_paths = []

        for config in configs_batch:
            os.makedirs("tmp", exist_ok=True)

            config_path = f"tmp/{str(uuid.uuid4())}-train.yaml"
            logging.debug(f"Writing a config path at {config_path} ...")
            configs_batch_paths.append(config_path)
            write_yaml(config, config_path)

        commands = [
            f"nohup python train.py --config {cp} > {cp.replace('yaml','log')}"
            for cp in configs_batch_paths
        ]

        procs = []
        for command in commands:
            logging.debug(f"running {command} ...")
            procs.append(Popen(command, shell=True))

        assert len(procs) == len(commands)
        for p, c in zip(procs, commands):
            logging.debug(f"waiting for the process {p}: {c} to be done ...")
            p.wait()
            logging.info(f"the process {p}: {c} is done!")

        logging.info(f"Running on {i+1} / {num_batches} batch is complete ...")

    logging.info("EVERYTHING DONE!")
