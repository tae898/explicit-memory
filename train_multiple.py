"""This script is to tran multiple train.py in parallel.
Things learned:
1. gamma=0.99 is always worse than gamma=0.65
"""
import os
import shutil
import subprocess
from copy import deepcopy

from tqdm import tqdm

from utils import read_yaml, write_yaml

train_config = read_yaml("./train.yaml")
commands = []
num_parallel = 2
reverse = True
shutil.rmtree("./junks", ignore_errors=True)
os.makedirs("./junks", exist_ok=False)
for capacity in [64]:
    for pretrain_semantic in [True, False]:
        for seed in [0, 1, 2, 3, 4]:
            train_config["capacity"] = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
            train_config["pretrain_semantic"] = pretrain_semantic
            train_config["seed"] = seed
            train_config["gpus"] = 1

            config_file_name = f"./junks/{capacity}_{pretrain_semantic}_{seed}.yaml"

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
