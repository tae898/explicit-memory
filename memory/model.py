import logging
from typing import Tuple, List

import os

from .memory import EpisodicMemory, SemanticMemory
from .environment.generator import OQAGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


eps = np.finfo(np.float32).eps.item()

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor) -> None:
    """Create sinusoidal embeddings, as in "Attention is All You Need".

    Copied from https://github.com/huggingface/transformers/blob/
    455c6390938a5c737fa63e78396cedae41e4e87e/src/transformers/modeling_distilbert.py#L53

    This is done in place.

    Args
    ----
    n_pos: number of positions (e.g., number of tokens)
    dim: size of the positional embedding vector (e.g., normally 768 is used for LMs. )
    out: torch weight to overwrite to.

    """
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()  # I don't think this is necessary?
    logging.info(f"Sinusodial embeddings are used {n_pos} * {dim}.")


class Embeddings(nn.Module):
    """Turn a given memory table into learanble embeddings."""

    def __init__(
        self,
        generator_params: dict,
        embedding_dim: int,
        num_rows: int,
        num_cols: int,
        special_tokens: dict = {"<pad>": 0, "<mask>": 1},
        device: str = "cpu",
        sinusoidal_pos_embds: bool = True,
    ) -> None:
        """Initialize an Embeddings object.

        Args
        ----
        generator_params: OQAGenerator parameters
        embedding_dim: embedding dimension (e.g., 4)
        num_rows: number of rows in the table.
        num_cols: number of columns in the table.
        special_tokens: dict = {"<pad>": 0, "<mask>": 1}
        device: "cpu" or "cuda"
        sinusoidal_pos_embds: True if you want to use sinusoidal, False if you want to
            use learnable.

        """
        super().__init__()
        oqag = OQAGenerator(**generator_params)
        self.embedding_dim = embedding_dim
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.heads = nn.Embedding(len(oqag.heads), self.embedding_dim)
        self.head2num = {head: idx for idx, head in enumerate(oqag.heads)}

        self.relations = nn.Embedding(len(oqag.relations), self.embedding_dim)
        self.relation2num = {
            relation: idx for idx, relation in enumerate(oqag.relations)
        }

        self.tails = nn.Embedding(len(oqag.tails), self.embedding_dim)
        self.tail2num = {tail: idx for idx, tail in enumerate(oqag.tails)}

        self.names = nn.Embedding(len(oqag.names), self.embedding_dim)
        self.name2num = {name: idx for idx, name in enumerate(oqag.names)}

        self.specials = nn.Embedding(
            len(special_tokens), self.embedding_dim, padding_idx=special_tokens["<pad>"]
        )
        self.special2num = {token: num for token, num in special_tokens.items()}

        self.positional_embeddings = nn.Embedding(num_rows, num_cols)

        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=num_rows,
                dim=num_cols,
                out=self.positional_embeddings.weight,
            )

        self.device = torch.device(device)

    def episodic2numerics(self, M_e: EpisodicMemory) -> torch.Tensor:
        """Convert the EpisodicMemory object into numerical values.

        Args
        ----
        M_e: EpisodicMemory object.

        Returns
        -------
        Episodic memory table whose shape is (num_memories, embedding_dim*3)

        """
        table = []
        for mem in M_e.entries:
            head = mem[0]
            relation = mem[1]
            tail = mem[2]
            # timestamps are not used at the moment since the memories are ordered by
            # time anyways.
            # timestamp = mem[3]

            name1, head = M_e.split_name_entity(head)
            name2, tail = M_e.split_name_entity(tail)

            assert name1 == name2

            head = self.names(
                torch.tensor(self.name2num[name1], device=self.device)
            ) + self.heads(torch.tensor(self.head2num[head], device=self.device))

            relation = self.relations(
                torch.tensor(self.relation2num[relation], device=self.device)
            )

            tail = self.names(
                torch.tensor(self.name2num[name2], device=self.device)
            ) + self.tails(torch.tensor(self.tail2num[tail], device=self.device))

            table.append(torch.cat([head, relation, tail]))

        table = torch.stack(table)

        return table

    def eq2numerics(self, question: list) -> torch.Tensor:
        """Convert the question into numerical values.

        Args
        ----
        question: [head, relation]. E.g., [Tae's laptop, AtLocation]

        Returns
        -------
        Episodic question whose shape is (1, embedding_dim*3)

        """
        head = question[0]
        name, head = EpisodicMemory.split_name_entity(head)
        relation = question[1]

        head = self.names(
            torch.tensor(self.name2num[name], device=self.device)
        ) + self.heads(torch.tensor(self.head2num[head], device=self.device))

        relation = self.relations(
            torch.tensor(self.relation2num[relation], device=self.device)
        )

        tail = self.names(
            torch.tensor(self.name2num[name], device=self.device)
        ) + self.specials(torch.tensor(self.special2num["<mask>"], device=self.device))

        table = torch.cat([head, relation, tail]).unsqueeze(0)

        return table

    def semantic2numerics(self, M_s: SemanticMemory) -> torch.Tensor:
        """Convert the SemanticMemory object into numerical values.

        Args
        ----
        M_s: SemanticMemory object.

        Returns
        -------
        Semantic memory table whose shape is (num_memories, embedding_dim*3)

        """
        table = []
        for mem in M_s.entries:
            head = mem[0]
            relation = mem[1]
            tail = mem[2]

            # num_gen are not used at the moment since the memories are ordered by
            # num_gen anyways.
            # num_gen = mem[3]

            head = self.heads(torch.tensor(self.head2num[head], device=self.device))

            relation = self.relations(
                torch.tensor(self.relation2num[relation], device=self.device)
            )

            tail = self.tails(torch.tensor(self.tail2num[tail], device=self.device))

            table.append(torch.cat([head, relation, tail]))

        table = torch.stack(table)

        return table

    def sq2numerics(self, question: list) -> torch.Tensor:
        """Convert the question into numerical values.

        Args
        ----
        question: [head, relation]. E.g., [Tae's laptop, AtLocation]

        Returns
        -------
        Semantic question whose shape is (1, embedding_dim*3)

        """
        head = SemanticMemory.remove_name(question[0])
        relation = question[1]

        head = self.heads(torch.tensor(self.head2num[head], device=self.device))

        relation = self.relations(
            torch.tensor(self.relation2num[relation], device=self.device)
        )

        tail = self.specials(
            torch.tensor(self.special2num["<mask>"], device=self.device)
        )

        table = torch.cat([head, relation, tail]).unsqueeze(0)

        return table

    def forward(
        self, M_e: EpisodicMemory, M_s: SemanticMemory, question: list, policy_type: str
    ):
        """One forward pass.

        Args
        ----
        M_e: EpisodicMemory object.
        M_s: SemanticMemory object.
        question: [head, relation]. E.g., [Tae's laptop, AtLocation]
        policy_type: one of the below:
            episodic_memory_manage
            episodic_question_answer
            semantic_memory_manage
            semantic_question_answer
            episodic_to_semantic
            episodic_semantic_question_answer

        """
        x = torch.zeros(
            self.num_rows, self.num_cols, dtype=torch.float32, device=self.device
        )

        if policy_type == "episodic_memory_manage":
            table = self.episodic2numerics(M_e)
            x[: table.shape[0], :] = table

        elif policy_type == "episodic_question_answer":
            table = self.episodic2numerics(M_e)
            x[: table.shape[0], :] = table

            table = self.eq2numerics(question)
            x[-1, :] = table

        elif policy_type == "semantic_memory_manage":
            table = self.semantic2numerics(M_s)
            x[: table.shape[0], :] = table

        elif policy_type == "semantic_question_answer":
            table = self.semantic2numerics(M_s)
            x[: table.shape[0], :] = table

            table = self.sq2numerics(question)
            x[-1, :] = table

        elif policy_type == "episodic_semantic_question_answer":
            table = self.episodic2numerics(M_e)
            if len(table) > 0:
                x[: table.shape[0], :] = table

            table = self.semantic2numerics(M_s)
            if len(table) > 0:
                x[M_e.capacity : M_e.capacity + table.shape[0], :] = table

            table = self.eq2numerics(question)
            x[-1, :] = table

        else:
            raise ValueError

        for idx in range(x.shape[0]):
            x[idx] += self.positional_embeddings(torch.tensor(idx, device=self.device))

        # x = x[torch.randperm(x.shape[0])]
        return x


class MLP(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        capacity: dict,
        policy_type: str,
        embedding_dim: int,
        generator_params: dict,
        dropout: float = 0.0,
    ) -> None:
        """Initialize an MLP object.

        Args
        ----
        capacity: memory capacity
        policy_type: one of the below:
            episodic_memory_manage
            episodic_question_answer
            semantic_memory_manage
            semantic_question_answer
            episodic_to_semantic
            episodic_semantic_question_answer
        embedding_dim: int,
        generator_params: dict,
        dropout: dropout probability

        """
        super().__init__()

        embeddings_params = {
            "generator_params": generator_params,
            "embedding_dim": embedding_dim,
        }
        self.policy_type = policy_type

        num_cols = embedding_dim * 3
        if policy_type == "episodic_memory_manage":
            num_rows = capacity["episodic"] + 1
            num_actions = capacity["episodic"] + 1
        elif policy_type == "episodic_question_answer":
            num_rows = capacity["episodic"] + 1
            num_actions = capacity["episodic"]
        elif policy_type == "semantic_memory_manage":
            num_rows = capacity["semantic"] + 1
            num_actions = capacity["semantic"] + 1
        elif policy_type == "semantic_question_answer":
            num_rows = capacity["semantic"] + 1
            num_actions = capacity["semantic"]
        elif policy_type == "episodic_semantic_question_answer":
            num_rows = capacity["episodic"] + capacity["semantic"] + 1
            num_actions = capacity["episodic"] + capacity["semantic"]
        else:
            raise ValueError

        embeddings_params["num_rows"] = num_rows
        embeddings_params["num_cols"] = num_cols

        self.embeddings = Embeddings(**embeddings_params)

        in_features = num_rows * num_cols
        self.in_features = in_features
        self.out_features = num_actions
        self.hidden_features = in_features

        # self.fc1 = nn.Linear(self.in_features, self.in_features)
        # self.fc2 = nn.Linear(self.in_features, self.out_features)
        # self.fc3 = nn.Linear(self.in_features, 1)

        self.affine1 = nn.Linear(self.in_features, self.hidden_features)

        # actor's layer
        self.action_head = nn.Linear(self.hidden_features, self.out_features)

        # critic's layer
        self.value_head = nn.Linear(self.hidden_features, 1)

        self.dropout = nn.Dropout(dropout)

        # things to cache for some RL algorithms.
        self.saved_actions = []

    def set_device(self, device: str = "cpu") -> None:
        """Send the model to CPU or GPU memory.

        Args
        ----
        device: either "cpu" or "cuda"

        """
        device = torch.device(device)
        self.embeddings.device = device
        self.to(device)

    def forward(
        self, M_e: EpisodicMemory, M_s: SemanticMemory, question: list
    ) -> torch.Tensor:
        """One forward pass.

        Args
        ----
        state: a numeric state whose shape is (1, in_features)

        """
        state = self.embeddings(M_e, M_s, question, self.policy_type)
        state = state.flatten()
        state = state.unsqueeze(0)

        x = F.relu(self.affine1(F.relu(self.affine1(state))))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


def create_policy_net(
    capacity: dict,
    policy_type: str,
    generator_params: dict,
    model_params: dict
) -> nn.Module:
    """Create policy neural networks.

    Args
    ----
    capacity: memory capacity
    policy_type: one of the below:
        episodic_memory_manage
        episodic_question_answer
        semantic_memory_manage
        semantic_question_answer
        episodic_to_semantic
        episodic_semantic_question_answer
    generator_params: dict,
    model_params: dict

    Returns
    -------
    An instantiated nn.Module object.

    """
    function_type = model_params.pop('function_type')
    if function_type.lower() == "mlp":
        model = MLP(**model_params)

        return model

    else:
        raise ValueError
