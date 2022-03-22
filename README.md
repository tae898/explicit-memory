# explicit-memory

This repo is to train an agent that has human-like memory systems. We explictly model it
with an explicit (i.e., semantic and episodic) memory system.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.8 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. `pip install -r requirements.txt`

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples
whose format is (`head`, `AtLocation`, `tail`). Here `head` is one of the 80 MS COCO
dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:

```
python collect_data.py
```

Otherwise, just use `./data/semantic-knowledge.json`

## The Room environment

We have released a challenging [OpenAI Gym](https://gym.openai.com/) compatible environment, ["the Room"](memory/environment/room.py). Check out our paper to have better idea what it's about.

## TODOs

1. Add unittests.
1. [Release the gym environment properly](https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai).
1. Better documentation.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make style && quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Authors

- [Taewoon Kim](https://taewoon.kim/)
