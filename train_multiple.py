"""This script is to tran multiple train.py in parallel.
Things learned:
1. gamma=0.99 is always worse than gamma=0.65
"""
import datetime
import os
import subprocess
from copy import deepcopy

from tqdm import tqdm

from utils import write_yaml

train_config = {
    "allow_random_human": False,
    "allow_random_question": True,
    "pretrain_semantic": False,
    "varying_rewards": False,
    "seed": 0,
    "num_eval_iter": 10,
    "max_epochs": 16,
    "batch_size": 1024,
    "epoch_length": 131072,
    "replay_size": 131072,
    "warm_start_size": 131072,
    "eps_end": 0,
    "eps_last_step": 2048,
    "eps_start": 1.0,
    "gamma": 0.65,
    "lr": 0.001,
    "sync_rate": 10,
    "loss_function": "huber",
    "optimizer": "adam",
    "des_size": "l",
    "des_version": "v2",
    "capacity": {"episodic": 16, "semantic": 16, "short": 1},
    "question_prob": 0.1,
    "observation_params": "perfect",
    "nn_params": {
        "architecture": "lstm",
        "embedding_dim": 32,
        "hidden_size": 64,
        "include_human": "sum",
        "memory_systems": ["episodic", "semantic", "short"],
        "num_layers": 2,
        "human_embedding_on_object_location": False,
    },
    "log_every_n_steps": 1,
    "early_stopping_patience": 1000,
    "precision": 32,
    "gpus": 0,
}

commands = []
num_parallel = 4
reverse = False
os.makedirs("./junks", exist_ok=True)

for capacity in [64]:
    for pretrain_semantic in [False]:
        for seed in [0, 1, 2, 3, 4]:
            train_config["capacity"] = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
            train_config["pretrain_semantic"] = pretrain_semantic
            train_config["seed"] = seed

            config_file_name = (
                f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"
            )

            write_yaml(train_config, config_file_name)

            commands.append(f"python train.py --config {config_file_name}")

for capacity in [16]:
    for pretrain_semantic in [True]:
        for seed in [2, 3, 4]:
            train_config["capacity"] = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
            train_config["pretrain_semantic"] = pretrain_semantic
            train_config["seed"] = seed

            config_file_name = (
                f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"
            )

            write_yaml(train_config, config_file_name)

            commands.append(f"python train.py --config {config_file_name}")

for capacity in [32]:
    for pretrain_semantic in [True]:
        for seed in [0, 1, 2, 3, 4]:
            train_config["capacity"] = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
            train_config["pretrain_semantic"] = pretrain_semantic
            train_config["seed"] = seed

            config_file_name = (
                f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"
            )

            write_yaml(train_config, config_file_name)

            commands.append(f"python train.py --config {config_file_name}")


for capacity in [64]:
    for pretrain_semantic in [True]:
        for seed in [2, 3, 4]:
            train_config["capacity"] = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
            train_config["pretrain_semantic"] = pretrain_semantic
            train_config["seed"] = seed

            config_file_name = (
                f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"
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


# allow_random_human=True-allow_random_question=True
# capacity=8, strategy=episodic,	pre_sem=False,	rewards_mean=-2.3, rewards_std=2.9
# capacity=8, strategy=semantic,	pre_sem=False,	rewards_mean=6.1, rewards_std=3.91
# capacity=8, strategy=random,	pre_sem=False,	rewards_mean=4.3, rewards_std=4.148
# capacity=8, strategy=pre_sem,	pre_sem=True,	rewards_mean=7.9, rewards_std=3.419
# ('des_size=s-capacity=8-allow_random_human=True-allow_random_question=True-pretrain=False-varying_rewards=False-gamma=0.65-gpus=0',
#   5.08),
#  ('des_size=s-capacity=8-allow_random_human=True-allow_random_question=True-pretrain=True-varying_rewards=False-gamma=0.65-gpus=0',
#   6.94)]

# allow_random_human=True-allow_random_question=False
# capacity=8, strategy=episodic,	pre_sem=False,	rewards_mean=4.3, rewards_std=3.348
# capacity=8, strategy=semantic,	pre_sem=False,	rewards_mean=8.7, rewards_std=4.818
# capacity=8, strategy=random,	pre_sem=False,	rewards_mean=5.5, rewards_std=5.162
# capacity=8, strategy=pre_sem,	pre_sem=True,	rewards_mean=9.3, rewards_std=4.818
#  ('des_size=s-capacity=8-allow_random_human=True-allow_random_question=False-pretrain=False-varying_rewards=False-gamma=0.65-gpus=0',
#   7.24),
#  ('des_size=s-capacity=8-allow_random_human=True-allow_random_question=False-pretrain=True-varying_rewards=False-gamma=0.65-gpus=0',
#   7.94),

# allow_random_human=False-allow_random_question=True
# capacity=8, strategy=episodic,	pre_sem=False,	rewards_mean=-1.8, rewards_std=2.4
# capacity=8, strategy=semantic,	pre_sem=False,	rewards_mean=7.2, rewards_std=3.156
# capacity=8, strategy=random,	pre_sem=False,	rewards_mean=2.6, rewards_std=3.929
# capacity=8, strategy=pre_sem,	pre_sem=True,	rewards_mean=7.4, rewards_std=3.72
#  ('des_size=s-capacity=8-allow_random_human=False-allow_random_question=True-pretrain=False-varying_rewards=False-gamma=0.65-gpus=0',
#   7.42),
#  ('des_size=s-capacity=8-allow_random_human=False-allow_random_question=True-pretrain=True-varying_rewards=False-gamma=0.65-gpus=0',
#   7.98),

# allow_random_human=False-allow_random_question=False
# capacity=8, strategy=episodic,	pre_sem=False,	rewards_mean=0.6, rewards_std=4.152
# capacity=8, strategy=semantic,	pre_sem=False,	rewards_mean=9.0, rewards_std=3.435
# capacity=8, strategy=random,	pre_sem=False,	rewards_mean=6.2, rewards_std=2.088
# capacity=8, strategy=pre_sem,	pre_sem=True,	rewards_mean=10.2, rewards_std=3.156
# [('des_size=s-capacity=8-allow_random_human=False-allow_random_question=False-pretrain=False-varying_rewards=False-gamma=0.65-gpus=0',
#   8.78),
#  ('des_size=s-capacity=8-allow_random_human=False-allow_random_question=False-pretrain=True-varying_rewards=False-gamma=0.65-gpus=0',
#   8.26),
