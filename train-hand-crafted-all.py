import subprocess
from utils import write_yaml
from tqdm import tqdm

config = {
    "memory_type": None,
    "policy": {"FIFO": None, "NRO": None},
    "save_at": "training-results",
    "data_path": "./data/data.json",
    "capacity": {"episodic": None, "semantic": None},
    "pretrained_semantic": None,
    "question_path": "./data/questions.json",
    "creation_time": 1602929100,
    "seed": 42,
}

# for capacity in tqdm(
#     [
#         (2, 0),
#         # (4, 0),
#         # (8, 0),
#         # (16, 0),
#         # (32, 0),
#         # (64, 0),
#         # (128, 0),
#         # (256, 0),
#         # (512, 0),
#         (1024, 0),
#     ]
# ):
#     for FIFO in tqdm([True, False]):
#         for NRO in tqdm([True, False]):

#             config["memory_type"] = "episodic"
#             config["capacity"]["episodic"] = capacity[0]
#             config["capacity"]["semantic"] = capacity[1]
#             config["policy"]["FIFO"] = FIFO
#             config["policy"]["NRO"] = NRO
#             config["pretrained_semantic"] = None
#             write_yaml(config, "./train-hand-crafted.yaml")
#             subprocess.run(["python", "train-hand-crafted.py"])

# for capacity in tqdm(
#     [
#         (2, 0),
#         # (4, 0),
#         # (8, 0),
#         # (16, 0),
#         # (32, 0),
#         # (64, 0),
#         # (128, 0),
#         # (256, 0),
#         # (512, 0),
#         (1024, 0),
#     ]
# ):
#     for FIFO in tqdm([True, False]):
#         for NRO in tqdm([True, False]):

#             config["memory_type"] = "semantic"
#             config["capacity"]["episodic"] = capacity[1]
#             config["capacity"]["semantic"] = capacity[0]
#             config["policy"]["FIFO"] = FIFO
#             config["policy"]["NRO"] = NRO
#             config["pretrained_semantic"] = None
#             write_yaml(config, "./train-hand-crafted.yaml")
#             subprocess.run(["python", "train-hand-crafted.py"])

# for capacity in tqdm([
#     (1, 1),
#     # (2, 2),
#     # (4, 4),
#     # (8, 8),
#     # (16, 16),
#     # (32, 32),
#     # (64, 64),
#     # (128, 128),
#     # (256, 256),
#     # (512, 512),
# ]):
#     for FIFO in tqdm([True, False]):
#         for NRO in tqdm([True, False]):

#             config["memory_type"] = "both"
#             config["capacity"]["episodic"] = capacity[1]
#             config["capacity"]["semantic"] = capacity[1]
#             config["policy"]["FIFO"] = FIFO
#             config["policy"]["NRO"] = NRO
#             config["pretrained_semantic"] = None
#             write_yaml(config, "./train-hand-crafted.yaml")
#             subprocess.run(["python", "train-hand-crafted.py"])

for capacity in tqdm([
    # (1, 1),
    # (2, 2),
    # (4, 4),
    # (8, 8),
    # (16, 16),
    # (32, 32),
    # (64, 64),
    # (128, 128),
    # (256, 256),
    (512, 512),
]):
    for FIFO in tqdm([True]):
        for NRO in tqdm([True]):
            config["memory_type"] = "both"
            config["capacity"]["episodic"] = capacity[1]
            config["capacity"]["semantic"] = capacity[1]
            config["policy"]["FIFO"] = FIFO
            config["policy"]["NRO"] = NRO
            config["pretrained_semantic"] = "./data/semantic-knowledge.json"
            write_yaml(config, "./train-hand-crafted.yaml")
            subprocess.run(["python", "train-hand-crafted.py"])
