# explicit-memory

This repo is to train an agent that has human-like memory systems. We explictly model it with an explicit (i.e., semantic and episodic) memory system.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.7 or higher. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples whose format is (`head`, `AtLocation`, `tail`). Here `head` is one of the 80 MS COCO dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:

```
python collect_data.py
```

Otherwise, just use `./data/data.json`

## Evaluation

We made the following five models (policies).

### (i) Hand-crafted 1: Only episodic, FIFO and NRO.

This agent only has an episodic memory system. Memory is maintained as FIFO (first in, first out) and NRO (new replaces old). FIFO is basically a way to forget old memories so that there is room to store new memories. NRO works in a way that if, for example, an old observation `<Karen's laptop,AtLocation,Karen's desk>}` is already in $\\bm{M}_{E}$ and there is a new incoming observation `<Karen's laptop,AtLocation,Karen's house>}`, then the new one replaces the old one. $\\bm{M}_{E}$ is empty in the beginning of an episode.

### (ii) Hand-crafted 2: Only semantic, FIFO and NRO.

This agent only has a semantic memory system. $\\bm{M}\_{S}$ is empty in the beginning of an episode.

### (iii) Hand-crafted 3: Both episodic and semantic, generalization, FIFO and NRO.

This agent has both episodic and semantic memory systems. The generalization is achieved by finding episodic memories that can be compressed into one semantic memory. Both $\\bm{M}_{E}$ and $\\bm{M}_{S}$ is empty in the beginning of an episode.

### (iv) RL 1: Both episodic and semantic, where semantic is scratch.

A reinforcement learning agent in an MDP environment learns how to generalize episodic memories into a semantic memory, and at the same time it learns how to forget redundant episodic and semantic memories. Both $\\bm{M}_{E}$ and $\\bm{M}_{S}$ is empty in the beginning of an episode.

### (v) RL 2: Both episodic and semantic, where semantic is pretrained.

A reinforcement learning agent in an MDP environment. The semantic memory system is pretrained from ConceptNet. $\\bm{M}_{E}$ is empty in the beginning and $\\bm{M}_{S}$ is populated with the commonsense knowledge. $\\bm{M}\_{S}$ does not change throughout training.

To be fair, the total memory capacities are the same for each model. Since our experiment is about learning how to organize memories, we abstract away the question answering part. This means that as long as the relevant memory is in the memory systems, then we assume that the agent answers the question correctly and that it gets a reward of $+1$. If it can't be found in them, then the reward is $0$.

## Troubleshooting

The best way to find and solve your problems is to see in the github issue tab. If you can't find what you want, feel free to raise an issue. We are pretty responsive.

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
