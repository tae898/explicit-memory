import argparse
import logging
import os
import time
from datetime import datetime
from subprocess import Popen

from tqdm import tqdm

from memory.utils import write_json

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

seeds = [0, 3, 6, 9, 12]
max_history = 1024
weighting_mode = "highest"
capacities = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
commonsense_prob = 0.5
semantic_knowledge_path = "./data/semantic-knowledge.json"
names_path = "./data/top-human-names"


def generate_all_configs():
    logging.debug("Generating configs ...")
    configs = []

    for capacity in capacities:
        for forget in [
            {"episodic": "oldest", "semantic": "weakest"},
            {"episodic": "random", "semantic": "random"},
        ]:
            for answer in [
                {"episodic": "latest", "semantic": "strongest"},
                {"episodic": "random", "semantic": "random"},
            ]:
                for memory_type in ["episodic", "semantic"]:
                    for seed in seeds:
                        config = {
                            "memory_type": memory_type,
                            "policy": {
                                "episodic": {"forget": None, "answer": None},
                                "semantic": {"forget": None, "answer": None},
                            },
                            "save_at": None,
                            "names_path": names_path,
                            "capacity": {"episodic": 0, "semantic": 0},
                            "semantic_knowledge_path": semantic_knowledge_path,
                            "pretrain_semantic": False,
                            "max_history": max_history,
                            "seed": seed,
                            "commonsense_prob": commonsense_prob,
                            "weighting_mode": weighting_mode,
                        }
                        config["capacity"][memory_type] = capacity * 2
                        config["policy"][memory_type]["forget"] = forget[memory_type]
                        config["policy"][memory_type]["answer"] = answer[memory_type]

                        configs.append(config)

    for capacity in capacities:
        for forget_semantic in ["weakest", "random"]:
            for answer_semantic in ["strongest", "random"]:
                for forget_episodic in ["oldest", "random"]:
                    for answer_episodic in ["latest", "random"]:
                        for pretrain_semantic in [True, False]:
                            for seed in seeds:
                                config = {
                                    "memory_type": "both",
                                    "policy": {
                                        "episodic": {"forget": None, "answer": None},
                                        "semantic": {"forget": None, "answer": None},
                                    },
                                    "save_at": None,
                                    "names_path": names_path,
                                    "capacity": {
                                        "episodic": capacity,
                                        "semantic": capacity,
                                    },
                                    "semantic_knowledge_path": semantic_knowledge_path,
                                    "pretrain_semantic": None,
                                    "max_history": max_history,
                                    "seed": seed,
                                    "commonsense_prob": commonsense_prob,
                                    "weighting_mode": weighting_mode,
                                }
                                config["policy"]["semantic"]["forget"] = forget_semantic
                                config["policy"]["semantic"]["answer"] = answer_semantic
                                config["policy"]["episodic"]["forget"] = forget_episodic
                                config["policy"]["episodic"]["answer"] = answer_episodic
                                config["pretrain_semantic"] = pretrain_semantic

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

            config_path = f"training-results/{current_time}/train-hand-crafted.json"
            logging.debug(f"Writing a config path at {config_path} ...")
            configs_batch_paths.append(config_path)
            write_json(config, config_path)

            time.sleep(1)

        commands = [
            f"nohup python train-hand-crafted.py --config {cp} > {cp.replace('json','log')}"
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
