# explicit-memory

For the documentation of [the older paper](https://arxiv.org/abs/2204.01611), check out
[this documentation](./v0/README-v0.md)

This repo is to train an agent that interacts with the [RoomEnv-v2](https://github.com/tae898/room-env).
See the [paper](todo/update/the/paper) for more information.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.8 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. `pip install -r requirements.txt`

## Training

```python
python train.py --config train.yaml
```

The hyperparameters can be configured in `train.yaml`. The training results with the
checkpoints will be saved at `./training_results/`

## Results

|                 Average loss, training.                 |           Average total rewards per episode, validation.           |              Average total rewards per episode, test.               |           Average total rewards, varying capacities, test.           |
| :-----------------------------------------------------: | :----------------------------------------------------------------: | :-----------------------------------------------------------------: | :------------------------------------------------------------------: |
| ![](./figures/des_size=l-capacity=32-train_loss-v1.svg) | ![](./figures/des_size=l-capacity=32-val_total_reward_mean-v1.svg) | ![](./figures/des_size=l-capacity=32-test_total_reward_mean-v1.svg) | ![](./figures/des_size=l-capacity=all-test_total_reward_mean-v1.svg) |

Also check out [`./models/`](./models) to see the saved training results. The `test_debug`
results might especially be interesting to you.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## [Cite our paper](todo/update/the/paper)

```bibtex
new paper bibtex coming soon
```

## Cite our code

[![DOI](https://zenodo.org/badge/411241603.svg)](https://zenodo.org/badge/latestdoi/411241603)

## Authors

- [Taewoon Kim](https://taewoon.kim/)
- [Michael Cochez](https://www.cochez.nl/)
- [Vincent Francois-Lavet](http://vincent.francois-l.be/)
- [Mark Neerincx](https://ocw.tudelft.nl/teachers/m_a_neerincx/)
- [Piek Vossen](https://vossen.info/)

## License

[MIT](https://choosealicense.com/licenses/mit/)
