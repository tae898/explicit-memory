# explicit-memory

This repo is to train an agent that has human-like memory systems. We explictly model it
with an explicit (i.e., semantic and episodic) memory system.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.7 or higher. 
2. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
4. `pip install -r requirements.txt`

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples
whose format is (`head`, `AtLocation`, `tail`). Here `head` is one of the 80 MS COCO
dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:

```
python collect_data.py
```

Otherwise, just use `./data/data.json`

## Evaluation

Using cognitive science and commonsense knowledge, we've come up with six startegies
(policies)

### (i) Episodic Memory Manage

Considering how the data is given, the best hand-crafted strategy is to remove the oldest
memory in the episodic memory system (upper bound).

A uniform-random strategy is to select and remove uniform-randomly selected memory (lower bound).

An RL agent has to learn its own strategy (policy). It doesn't have to be as good as the
hand-crafted one, but I still expect some promising results.

State: $N$ episodic memories + $1$ incoming episodic memory. This results in $N+1$ rows\
Action: $N+1$ discrete actions. For example, if the action is $k$, its removing $k$ th memory.\
Reward: If the episodic QA answers the question properly, then it's $+1$ and otherwise it's $+0$

### (ii) Episodic Question Answer

Considering how the data is given, the best hand-crafted strategy is first to find the
episodic memories whose head is the same as quesiton query head. And then among the $M$
selected memories, it selects the latest one. (upper bound)

A uniform-random strategy is to select and remove uniform-randomly selected memory (lower bound).

An RL agent has to learn its own strategy (policy). It doesn't have to be as good as the
hand-crafted one, but I still expect some promising results

State: $N$ episodic memories + $1$ episodic question query. This results in $N+1$ rows\
Action: $N+1$ discrete actions. For example, if the action is $k$, its selecting $k$ th
memory to answer the question.\
Reward: If retrieved the memory's tail is the correct location of the object, then $+0$,
otherwise it's $0$.

### (iii) Semantic Memory Manage

Considering how the data is given, the best hand-crafted strategy is to remove the weakest
memory in the semantic memory system (upper bound).

A uniform-random strategy is to select and remove uniform-randomly selected memory (lower bound).

An RL agent has to learn its own strategy (policy). It doesn't have to be as good as the
hand-crafted one, but I still expect some promising results.

State: $N$ semantic memories + $1$ incoming semantic memory. This results in $N+1$ rows\
Action: $N+1$ discrete actions. For example, if the action is $k$, its removing $k$ th memory.\
Reward: If the semantic QA answers the question properly, then it's $+1$ and otherwise it's $+0$

### (iv) Semantic Question Answer

Considering how the data is given, the best hand-crafted strategy is first to find the
semantic memories whose head is the same as quesiton query head. And then among the $M$
selected memories, it selects the strongest one. (upper bound)

A uniform-random strategy is to select and remove uniform-randomly selected memory (lower bound).

An RL agent has to learn its own strategy (policy). It doesn't have to be as good as the
hand-crafted one, but I still expect some promising results.

State: $N$ semantic memories + $1$ semantic question query. This results in $N+1$ rows\
Action: $N+1$ discrete actions. For example, if the action is $k$, its selecting $k$ th
memory. to answer the question.\
Reward: If retrieved the memory's tail is the correct location of the object, then $+0$,
otherwise it's $0$.

### (v): Episodic to Semantic

This only applies when the agent has both episodic and semantic memory systems.

Considering how the data is given, the best hand-crafted strategy is first to find the
$L \\geq 2$ episodic memories that are "similar" and then compress them to one semantic memory.
The similarity is defined as the episodic memories that have the same head and tail, if the
person's name is removed.

A uniform-random strategy is to select one episodic memory and turn it into a semantic
memory (lower bound).

An RL agent has to learn its own strategy (policy). It doesn't have to be as good as the
hand-crafted one, but I still expect some promising results.

State: $N$ episodic memories + $1$ incoming episodic memory. This results in $N+1$ rows\
Action: ?
Reward: ?

I'm actually not sure if RL makes sense here. Perhaps it's better to do some ontology
engineering + GNN to achieve compression.

### (vi): Episodic Semantic Question Answer

This only applies when the agent has both episodic and semantic memory systems.

Considering how the data is given, the best hand-crafted strategy is first try to answer
the question using the Episodic Question Answer and then Semantic Question Answer.

A uniform-random strategy is to select one memory from episodic and semantic memories
combined and use its tail to answer the question query.

An RL agent has to learn its own strategy (policy). It doesn't have to be as good as the
hand-crafted one, but I still expect some promising results.

State: $N\_{e}$ episodic memories + $N\_{m}$ semantic memories + $1$ incoming episodic memory.
This results in $N\_{1} + N\_{2} +1$ rows\
Action: $N\_{1} + N\_{2} +1$ discrete actions. For example, if the action is $k$, its
selecting $k$ th memory to answer the question.\
Reward: If retrieved the memory's tail is the correct location of the object, then $+0$,
otherwise it's $0$.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make style && quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Authors

- [Taewoon Kim](https://taewoonkim.com/)
