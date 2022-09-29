"""This script is to tran multiple train.py in parallel."""
import os
import subprocess
from copy import deepcopy

from tqdm import tqdm

from utils import read_yaml, write_yaml

train_config = read_yaml("./train.yaml")
commands = []
num_parallel = 8
reverse = False
os.makedirs("./junks", exist_ok=True)
for allow_random_human in [True, False]:
    for allow_random_question in [True, False]:
        for pretrain_semantic in [True, False]:
            for varying_rewards in [True, False]:
                for gamma in [0.65, 0.99]:
                    for seed in [0, 1, 2, 3, 4]:
                        train_config["allow_random_human"] = allow_random_human
                        train_config["allow_random_question"] = allow_random_question
                        train_config["pretrain_semantic"] = pretrain_semantic
                        train_config["varying_rewards"] = varying_rewards
                        train_config["gamma"] = gamma
                        train_config["seed"] = seed

                        config_file_name = (
                            f"./junks/{allow_random_human}_"
                            f"{allow_random_question}_{pretrain_semantic}_"
                            f"{varying_rewards}_{gamma}_{seed}.yaml"
                        )

                        write_yaml(train_config, config_file_name)

                        commands.append(f"python train.py --config {config_file_name}")

print(f"Running {len(commands)} training scripts ...")
if reverse:
    commands.reverse()
commands_original = deepcopy(commands)

commands_batched = [
    [commands[i * num_parallel + j] for j in range(num_parallel)]
    for i in range(len(commands) // num_parallel)
]

if len(commands) % num_parallel != 0:
    commands_batched.append(commands[-(len(commands) % num_parallel) :])

assert commands == [bar for foo in commands_batched for bar in foo]


for commands in tqdm(commands_batched):
    procs = [subprocess.Popen(command, shell=True) for command in commands]
    for p in procs:
        p.communicate()
