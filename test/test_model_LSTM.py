import unittest

import gym
import room_env
import torch

from model import LSTM


class LSTMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = gym.make("RoomEnv-v1", seed=42, des_size="xxs")

        self.dqn_none = LSTM(
            hidden_size=8,
            num_layers=1,
            n_actions=2,
            embedding_dim=4,
            capacity={"episodic": 1, "semantic": 1, "short": 2},
            des=self.env.des,
            include_human=None,
            memory_systems=["episodic", "semantic", "short"],
        )
        self.dqn_none.create_embeddings()

        self.dqn_sum = LSTM(
            hidden_size=8,
            num_layers=1,
            n_actions=2,
            embedding_dim=4,
            capacity={"episodic": 1, "semantic": 1, "short": 2},
            des=self.env.des,
            include_human="sum",
            memory_systems=["episodic", "semantic", "short"],
        )
        self.dqn_sum.create_embeddings()

        self.dqn_concat = LSTM(
            hidden_size=8,
            num_layers=1,
            n_actions=2,
            embedding_dim=4,
            capacity={"episodic": 1, "semantic": 1, "short": 2},
            des=self.env.des,
            include_human="concat",
            memory_systems=["episodic", "semantic", "short"],
        )
        self.dqn_concat.create_embeddings()

        self.dqn_sum_short = LSTM(
            hidden_size=8,
            num_layers=1,
            n_actions=2,
            embedding_dim=4,
            capacity={"episodic": 1, "semantic": 1, "short": 2},
            des=self.env.des,
            include_human="sum",
            memory_systems=["short"],
        )
        self.dqn_sum_short.create_embeddings()

    def test_short_memory_systems(self) -> None:
        pass

    def test_init_des(self) -> None:
        self.assertEqual(
            self.env.des.humans, ["Beverly", "Brittany", "Patricia", "Roger"]
        )
        self.assertEqual(self.env.des.human_locations, ["A"])
        self.assertEqual(self.env.des.objects, ["bowl"])
        self.assertEqual(self.env.des.object_locations, ["air", "cupboard"])

    def test_create_embeddings_word2idx(self) -> None:
        word2idx = {
            "<PAD>": 0,
            "Beverly": 1,
            "Brittany": 2,
            "Patricia": 3,
            "Roger": 4,
            "A": 5,
            "bowl": 6,
            "air": 7,
            "cupboard": 8,
        }

        self.assertEqual(self.dqn_none.word2idx, word2idx)
        self.assertEqual(self.dqn_sum.word2idx, word2idx)
        self.assertEqual(self.dqn_concat.word2idx, word2idx)

    def test_create_embeddings_embeddings(self) -> None:

        for dqn in [self.dqn_none, self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            self.assertEqual(dqn.embeddings.num_embeddings, len(dqn.word2idx))
            self.assertEqual(dqn.embeddings.padding_idx, 0)
            self.assertEqual(
                dqn.embeddings(
                    torch.randint(low=0, high=dqn.embeddings.num_embeddings, size=(1,))
                ).shape,
                torch.Size([1, dqn.embedding_dim]),
            )

    def test_make_embedding_padding_short(self) -> None:

        for dqn in [self.dqn_none, self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            emb = dqn.make_embedding(
                mem={
                    "human": "<PAD>",
                    "human_location": "<PAD>",
                    "object": "<PAD>",
                    "object_location": "<PAD>",
                    "timestamp": "<PAD>",
                },
                memory_type="short",
            )
            self.assertAlmostEqual((emb**2).sum().item(), 0)

    def test_make_embedding_padding_episodic(self) -> None:

        for dqn in [self.dqn_none, self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            emb = dqn.make_embedding(
                mem={
                    "human": "<PAD>",
                    "human_location": "<PAD>",
                    "object": "<PAD>",
                    "object_location": "<PAD>",
                    "timestamp": "<PAD>",
                },
                memory_type="episodic",
            )
            self.assertAlmostEqual((emb**2).sum().item(), 0)

    def test_make_embedding_padding_semantic(self) -> None:

        for dqn in [self.dqn_none, self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            emb = dqn.make_embedding(
                mem={
                    "object": "<PAD>",
                    "object_location": "<PAD>",
                    "num_generalized": "<PAD>",
                },
                memory_type="semantic",
            )
            self.assertAlmostEqual((emb**2).sum().item(), 0)

    def test_make_embedding_episodic(self) -> None:

        for dqn in [self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            emb = dqn.make_embedding(
                mem={
                    "human": "Beverly",
                    "human_location": "A",
                    "object": "<PAD>",
                    "object_location": "<PAD>",
                    "timestamp": 1234,
                },
                memory_type="episodic",
            )
            self.assertNotAlmostEqual((emb**2).sum().item(), 0)
            self.assertEqual(emb.shape, torch.Size([dqn.input_size_e]))

        emb = self.dqn_none.make_embedding(
            mem={
                "human": "Beverly",
                "human_location": "A",
                "object": "<PAD>",
                "object_location": "<PAD>",
                "timestamp": 1234,
            },
            memory_type="episodic",
        )
        self.assertAlmostEqual((emb**2).sum().item(), 0)
        self.assertEqual(emb.shape, torch.Size([self.dqn_none.input_size_e]))

    def test_make_embedding_semantic(self) -> None:

        for dqn in [self.dqn_none, self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            emb = dqn.make_embedding(
                mem={
                    "object": "<PAD>",
                    "object_location": "<PAD>",
                    "num_generalized": 33,
                },
                memory_type="semantic",
            )
            self.assertAlmostEqual((emb**2).sum().item(), 0)
            self.assertEqual(emb.shape, torch.Size([dqn.input_size_s]))

    def test_make_embedding_sum(self) -> None:
        emb_without_human = self.dqn_sum.make_embedding(
            mem={
                "human": "<PAD>",
                "human_location": "A",
                "object": "bowl",
                "object_location": "air",
                "timestamp": 0,
            },
            memory_type="short",
        )
        emb_with_human = self.dqn_sum.make_embedding(
            mem={
                "human": "Beverly",
                "human_location": "A",
                "object": "bowl",
                "object_location": "air",
                "timestamp": 0,
            },
            memory_type="episodic",
        )
        emb_expected = (
            self.dqn_sum.embeddings(
                torch.tensor(self.dqn_sum.word2idx["Beverly"])
            ).repeat(2)
            + emb_without_human
        )
        self.assertAlmostEqual(((emb_expected - emb_with_human) ** 2).sum().item(), 0)

    def test_make_embedding_concat(self) -> None:
        emb_without_human = self.dqn_concat.make_embedding(
            mem={
                "human": "<PAD>",
                "human_location": "A",
                "object": "bowl",
                "object_location": "air",
                "timestamp": 0,
            },
            memory_type="episodic",
        )
        self.assertAlmostEqual(
            (emb_without_human[: self.dqn_concat.embedding_dim] ** 2).sum().item(), 0
        )
        self.assertEqual(
            emb_without_human.shape, torch.Size([self.dqn_concat.input_size_e])
        )

        emb_without_human = self.dqn_concat.make_embedding(
            mem={
                "human": "Beverly",
                "human_location": "A",
                "object": "bowl",
                "object_location": "air",
                "timestamp": 0,
            },
            memory_type="short",
        )
        self.assertNotAlmostEqual(
            (emb_without_human[: self.dqn_concat.embedding_dim] ** 2).sum().item(), 0
        )
        self.assertEqual(
            emb_without_human.shape, torch.Size([self.dqn_concat.input_size_e])
        )

    def test_create_batch_semantic0(self) -> None:

        for dqn in [self.dqn_none, self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            x = [
                str(
                    [
                        {
                            "object": "bowl",
                            "object_location": "cupboard",
                            "num_generalized": 3,
                        }
                    ]
                ),
                str(
                    [
                        {
                            "object": "bowl",
                            "object_location": "air",
                            "num_generalized": 123,
                        }
                    ]
                ),
            ]
            batch = dqn.create_batch(x, 4, memory_type="semantic")
            self.assertEqual(batch.shape, torch.Size([2, 4, dqn.input_size_s]))

    def test_create_batch_short(self) -> None:

        x = [
            str(
                [
                    {
                        "human": "Beverly",
                        "human_location": "A",
                        "object": "bowl",
                        "object_location": "air",
                        "timestamp": 0,
                    },
                    {
                        "human": "Patricia",
                        "human_location": "A",
                        "object": "bowl",
                        "object_location": "cupboard",
                        "timestamp": 0,
                    },
                ]
            ),
            str(
                [
                    {
                        "human": "Roger",
                        "human_location": "A",
                        "object": "bowl",
                        "object_location": "cupboard",
                        "timestamp": 0,
                    }
                ]
            ),
        ]
        batch_non_human = self.dqn_none.create_batch(x, 4, memory_type="short")
        padding_indices = [[0, 2], [0, 3], [1, 1], [1, 2], [1, 3]]

        for indices in padding_indices:
            batch_idx = indices[0]
            row_idx = indices[1]
            self.assertAlmostEqual(
                (batch_non_human[batch_idx][row_idx] ** 2).sum().item(), 0
            )

        batch_human = self.dqn_sum.create_batch(x, 4, memory_type="episodic")
        padding_indices = [[0, 2], [0, 3], [1, 1], [1, 2], [1, 3]]

        for indices in padding_indices:
            batch_idx = indices[0]
            row_idx = indices[1]
            self.assertAlmostEqual(
                (batch_human[batch_idx][row_idx] ** 2).sum().item(), 0
            )

        non_padding_indices = [[0, 0], [0, 1], [1, 0]]

        for indices in non_padding_indices:
            batch_idx = indices[0]
            row_idx = indices[1]
            self.assertNotAlmostEqual(
                (
                    (
                        batch_human[batch_idx][row_idx]
                        - batch_non_human[batch_idx][row_idx]
                    )
                    ** 2
                )
                .sum()
                .item(),
                0,
            )

    def test_forward(self) -> None:

        # Here, we test the batch_size=4, des_size=xxs
        x = []
        # episodic
        x.append(
            (
                "[{'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 586}, {'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 587}]",
                "[{'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 311}, {'human': 'Roger', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 312}]",
                "[{'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 3}, {'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 6}]",
                "[{'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 808}, {'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 810}]",
            )
        )
        # semantic
        x.append(
            (
                "[{'object': 'bowl', 'object_location': 'cupboard', 'num_generalized': 1}, {'object': 'bowl', 'object_location': 'air', 'num_generalized': 77}]",
                "[{'object': 'bowl', 'object_location': 'cupboard', 'num_generalized': 1}, {'object': 'bowl', 'object_location': 'air', 'num_generalized': 45}]",
                "[{'object': 'bowl', 'object_location': 'cupboard', 'num_generalized': 1}, {'object': 'bowl', 'object_location': 'air', 'num_generalized': 3}]",
                "[{'object': 'bowl', 'object_location': 'cupboard', 'num_generalized': 1}, {'object': 'bowl', 'object_location': 'air', 'num_generalized': 102}]",
            )
        )
        # short
        x.append(
            (
                "[{'human': 'Beverly', 'human_location': 'A', 'object': 'bowl', 'object_location': 'air', 'timestamp': 588}, {'human': 'Roger', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 589}, {'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 590}, {'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 591}]",
                "[{'human': 'Beverly', 'human_location': 'A', 'object': 'bowl', 'object_location': 'air', 'timestamp': 314}, {'human': 'Roger', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 315}, {'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 316}, {'human': 'Roger', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 317}]",
                "[{'human': 'Beverly', 'human_location': 'A', 'object': 'bowl', 'object_location': 'air', 'timestamp': 8}, {'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 9}, {'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 10}, {'human': 'Roger', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 11}]",
                "[{'human': 'Beverly', 'human_location': 'A', 'object': 'bowl', 'object_location': 'air', 'timestamp': 811}, {'human': 'Roger', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 812}, {'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 813}, {'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 814}]",
            )
        )

        for dqn in [self.dqn_none, self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            qvalues = dqn(x)
            self.assertEqual(qvalues.shape, torch.Size([4, 2]))

        # Here, we test the batch_size=1, des_size=xxs
        x = []
        # episodic
        x.append(
            "[{'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 586}, {'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 587}]",
        )
        # semantic
        x.append(
            "[{'object': 'bowl', 'object_location': 'cupboard', 'num_generalized': 1}, {'object': 'bowl', 'object_location': 'air', 'num_generalized': 77}]",
        )
        # short
        x.append(
            "[{'human': 'Beverly', 'human_location': 'A', 'object': 'bowl', 'object_location': 'air', 'timestamp': 588}, {'human': 'Roger', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 589}, {'human': 'Brittany', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 590}, {'human': 'Patricia', 'human_location': 'A', 'object': 'bowl', 'object_location': 'cupboard', 'timestamp': 591}]",
        )

        for dqn in [self.dqn_none, self.dqn_sum, self.dqn_concat, self.dqn_sum_short]:
            qvalues = dqn(x)
            self.assertEqual(qvalues.shape, torch.Size([1, 2]))
