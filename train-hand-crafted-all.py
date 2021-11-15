import argparse
import time
from subprocess import Popen
from tqdm import tqdm
from utils import write_yaml
import logging
import os
from datetime import datetime

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def generate_all_configs():
    logging.debug("Generating configs ...")
    configs = []

    capacities = [
        (2, 0),
        (4, 0),
        (8, 0),
        (16, 0),
        (32, 0),
        (64, 0),
        (128, 0),
        (256, 0),
        (512, 0),
        (1024, 0),
    ]

    for capacity in capacities:
        for forget in ["oldest", "random"]:
            for answer in ["latest", "random"]:

                config = {
                    "memory_type": None,
                    "policy": {
                        "episodic": {"forget": None, "answer": None},
                        "semantic": {"forget": None, "answer": None},
                    },
                    "save_at": None,
                    "data_path": "./data/data.json",
                    "capacity": {"episodic": None, "semantic": None},
                    "pretrained_semantic": None,
                    "question_path": "./data/questions.json",
                    "seed": 42,
                }
                config["memory_type"] = "episodic"
                config["capacity"]["episodic"] = capacity[0]
                config["capacity"]["semantic"] = capacity[1]
                config["policy"]["episodic"]["forget"] = forget
                config["policy"]["episodic"]["answer"] = answer
                config["pretrained_semantic"] = None

                configs.append(config)

    capacities = [
        (2, 0),
        (4, 0),
        (8, 0),
        (16, 0),
        (32, 0),
        (64, 0),
        (128, 0),
        (256, 0),
        (512, 0),
        (1024, 0),
    ]
    for capacity in capacities:
        for forget in ["weakest", "random"]:
            for answer in ["strongest", "random"]:

                config = {
                    "memory_type": None,
                    "policy": {
                        "episodic": {"forget": None, "answer": None},
                        "semantic": {"forget": None, "answer": None},
                    },
                    "save_at": None,
                    "data_path": "./data/data.json",
                    "capacity": {"episodic": None, "semantic": None},
                    "pretrained_semantic": None,
                    "question_path": "./data/questions.json",
                    "seed": 42,
                }
                config["memory_type"] = "semantic"
                config["capacity"]["episodic"] = capacity[1]
                config["capacity"]["semantic"] = capacity[0]
                config["policy"]["semantic"]["forget"] = forget
                config["policy"]["semantic"]["answer"] = answer
                config["pretrained_semantic"] = None

                configs.append(config)

    capacities = [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ]
    for capacity in capacities:
        for forget_semantic in ["weakest", "random"]:
            for answer_semantic in ["strongest", "random"]:
                for forget_episodic in ["oldest", "random"]:
                    for answer_episodic in ["latest", "random"]:

                        config = {
                            "memory_type": None,
                            "policy": {
                                "episodic": {"forget": None, "answer": None},
                                "semantic": {"forget": None, "answer": None},
                            },
                            "save_at": None,
                            "data_path": "./data/data.json",
                            "capacity": {"episodic": None, "semantic": None},
                            "pretrained_semantic": None,
                            "question_path": "./data/questions.json",
                            "seed": 42,
                        }

                        config["memory_type"] = "both"
                        config["capacity"]["episodic"] = capacity[0]
                        config["capacity"]["semantic"] = capacity[1]
                        config["policy"]["semantic"]["forget"] = forget_semantic
                        config["policy"]["semantic"]["answer"] = answer_semantic
                        config["policy"]["episodic"]["forget"] = forget_episodic
                        config["policy"]["episodic"]["answer"] = answer_episodic
                        config["pretrained_semantic"] = None

                        configs.append(config)

    capacities = [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ]
    for capacity in capacities:
        for forget_semantic in ["weakest", "random"]:
            for answer_semantic in ["strongest", "random"]:
                for forget_episodic in ["oldest", "random"]:
                    for answer_episodic in ["latest", "random"]:

                        config = {
                            "memory_type": None,
                            "policy": {
                                "episodic": {"forget": None, "answer": None},
                                "semantic": {"forget": None, "answer": None},
                            },
                            "save_at": None,
                            "data_path": "./data/data.json",
                            "capacity": {"episodic": None, "semantic": None},
                            "pretrained_semantic": None,
                            "question_path": "./data/questions.json",
                            "seed": 42,
                        }

                        config["memory_type"] = "both"
                        config["capacity"]["episodic"] = capacity[0]
                        config["capacity"]["semantic"] = capacity[1]
                        config["policy"]["semantic"]["forget"] = forget_semantic
                        config["policy"]["semantic"]["answer"] = answer_semantic
                        config["policy"]["episodic"]["forget"] = forget_episodic
                        config["policy"]["episodic"]["answer"] = answer_episodic
                        config["pretrained_semantic"] = "./data/semantic-knowledge.json"

                        configs.append(config)

    logging.info(f"In total of {len(configs)} configs generated!")
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train ALL handcrafted policies")
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
            current_time = datetime.now().strftime(r"%m%d_%H%M%S")
            config["save_at"] = f"training-results/{current_time}"
            logging.debug(f"Creating a directory at {config['save_at']} ...")
            os.makedirs(config["save_at"], exist_ok=True)

            config_path = f"training-results/{current_time}/train-hand-crafted.yaml"
            logging.debug(f"Writing a config path at {config_path} ...")
            configs_batch_paths.append(config_path)
            write_yaml(config, config_path)

            time.sleep(1)

        commands = [
            f"nohup python train-hand-crafted.py --config {cp} > {cp.replace('yaml','log')}"
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
