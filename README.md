# explicit-memory

This repo is to train an agent that has human-like memory systems. We explictly model it with an explicit (i.e., semantic and episodic) memory system.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.7 or higher. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples whose format is (`head`, `AtLocation`, `tail`). Here `head` is one of the 80 MS COCO dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:
```
python collect-data.py
```

## Evaluation

To be fair, all of the below models should have the same amount of memory capacity.

### Hand-crafted 1: Only episodic, FIFO

This agent only has an episodic memory system. Memory is maintained as FIFO (first in first out).

### Hand-crafted 2: Only semantic, FIFO

This agent only has an semantic memory system. Memory is maintained as FIFO (first in first out).

### Hand-crafted 3: Both episodic and semantic, generalization + FIFO

This agent has both episodic and semantic memory systems. Memory is maintained as generalization and FIFO (first in first out).

### RL: Both episodic and semantic, where semantic is scratch

An reinforcement learning agent in an MDP environment. The semantic memory system is learned from scratch.

### RL: Both episodic and semantic, where semantic is pretrained

An reinforcement learning agent in an MDP environment. The semantic memory system is pretrained from ConceptNet.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

* [Taewoon Kim](https://taewoonkim.com/) 