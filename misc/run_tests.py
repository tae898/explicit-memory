import random

from tqdm import tqdm

from memory import EpisodicMemory, Memory, SemanticMemory
from memory.environment.generator import OQAGenerator
from memory.environment.gym import MemoryEnv

print("testing the generator ...")

gen_params = {
    "max_history": 1024,
    "semantic_knowledge_path": "./data/semantic-knowledge.json",
    "names_path": "./data/top-human-names",
    "weighting_mode": "highest",
    "commonsense_prob": 0.5,
    "time_start_at": 0,
    "limits": {
        "heads": 40,
        "tails": 1,
        "names": 20,
        "allow_spaces": False,
    },
    "disjoint_entities": True,
}

oqag = OQAGenerator(**gen_params)
oqag_ = OQAGenerator(**gen_params)

assert oqag == oqag_
assert len(oqag.heads) == gen_params["limits"]["heads"]
assert len(oqag.tails) <= gen_params["limits"]["heads"]
assert len(oqag.names) == gen_params["limits"]["names"]

for _ in tqdm(range(gen_params["max_history"])):
    ob, qa = oqag.generate_with_history()

assert oqag.is_full
oqag.reset()

rand_int = random.randint(1, gen_params["max_history"])
for _ in tqdm(range(rand_int)):

    ob, qa = oqag.generate_with_history()
    # import pdb; pdb.set_trace()
    assert oqag.is_possible_observation(ob)
    assert oqag.is_possible_qa(qa)

assert len(oqag.history) == rand_int
assert oqag.timestamp == gen_params["time_start_at"] + rand_int
oqag.clear_history()
assert len(oqag.history) == 0

print("testing the environment ...")

env = MemoryEnv(**gen_params)
state = env.reset()

step_count = 0
ep_rewards = 0
while True:
    action = EpisodicMemory.remove_name(state["observation"][2])
    state, reward, done, info = env.step(action)
    ep_rewards += reward
    step_count += 1

    if done:
        break

print(f"One episode is done. episode rewards: {ep_rewards}")
assert step_count == gen_params["max_history"]


print("testing the memory class ...")

capacity = {
    "episodic": random.randint(0, gen_params["max_history"]),
    "semantic": random.randint(0, gen_params["max_history"]),
}

M = Memory("episodic", 0)
assert not M.is_answerable(["Tae's laptop", "AtLocation", "desk"])

M_e = EpisodicMemory(capacity["episodic"])
M_e_ = EpisodicMemory(capacity["episodic"])

assert M_e == M_e_

M_s = SemanticMemory(capacity["semantic"])
M_s_ = SemanticMemory(capacity["semantic"])

assert M_s == M_s_

name_removed = M.remove_name("Tae's laptop")
assert "Tae's" not in name_removed
mem_epi = M_e.ob2epi(state["observation"])
entry_without_timestamp = M_e.remove_timestamp(state["observation"])
name, entity = M_e.split_name_entity("Tae's laptop")
assert name == "Tae"
assert entity == "laptop"

mem_sem = M_s.ob2sem(state["observation"])
M_s.clean_memories()


state = env.reset()
step_count = 0
ep_rewards = 0
for _ in range(random.randint(1, gen_params["max_history"])):
    action = Memory.remove_name(state["observation"][2])
    state, reward, done, info = env.step(action)
    ep_rewards += reward
    step_count += 1

    if not M_e.is_kinda_full:
        M_e.add(M_e.ob2epi(state["observation"]))

    if not M_s.is_kinda_full:
        M_s.add(M_s.ob2sem(state["observation"]))

timestamps = [mem[-1] for mem in M_e.entries]
assert min(timestamps) == 1
if M_e.is_kinda_full:
    assert max(timestamps) == M_e.capacity + 1
assert sorted(timestamps) == timestamps

numgens = [mem[-1] for mem in M_s.entries]
assert min(numgens) == 1
assert sorted(numgens) == numgens

M_s.forget_all()
M_s.add(M_s.ob2sem(["Tae's laptop", "AtLocation", "Tae's desk", 1]))
M_s.add(M_s.ob2sem(["Tae's laptop", "AtLocation", "Tae's desk", 2]))

assert M_s.size == 1
assert M_s.entries[0][3] == 2


gen_params = {
    "max_history": 1024,
    "semantic_knowledge_path": "./data/semantic-knowledge.json",
    "names_path": "./data/top-human-names",
    "weighting_mode": "weighted",
    "commonsense_prob": 0.5,
    "time_start_at": 0,
    "limits": {
        "heads": None,
        "tails": None,
        "names": None,
        "allow_spaces": True,
    },
    "disjoint_entities": True,
}

oqag = OQAGenerator(**gen_params)


assert len(oqag.heads) == 66
assert len(oqag.tails) == 485
assert len(oqag.names) == 20

print("ALL TESTS SUCCESSFULLY PASSED")
