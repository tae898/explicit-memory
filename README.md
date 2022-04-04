# explicit-memory

This repo is to train an agent that has human-like memory systems. We explictly model it
with an explicit (i.e., semantic and episodic) memory system.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.8 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. `pip install -r requirements.txt`

## The Room environment

We have released a challenging [OpenAI Gym](https://gym.openai.com/) compatible environment, ["the Room"](https://github.com/tae898/room-env).

## Heuristics

We provide some heuristics that can maximize the rewards. For the details, take a look at this paper.

## TODOs

1. Add unittests.
1. Better documentation.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Cite our work

[![DOI](https://zenodo.org/badge/411241603.svg)](https://zenodo.org/badge/latestdoi/411241603)

## Authors

- [Taewoon Kim](https://taewoon.kim/)
