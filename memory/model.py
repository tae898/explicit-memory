import logging
import os
from typing import Tuple

import torch
from torch import nn

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self, num_rows_cols: Tuple[int, int], num_actions: int):
        """
        Args
        ----
        num_rows_cols: (number of rows, number of cols)
        num_actions: number of discrete actions available in the environment

        """
        super().__init__()
        self.num_rows = num_rows_cols[0]
        self.num_cols = num_rows_cols[1]
        self.num_actions = num_actions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.bn_row = nn.BatchNorm1d(self.num_cols)
        self.bn_col = nn.BatchNorm1d(self.num_rows)
        self.dropout = nn.Dropout(0.5)

        self.LinearRow1 = nn.Linear(self.num_cols, self.num_cols)
        self.LinearRow2 = nn.Linear(self.num_cols, 1)

        self.LinearCol1 = nn.Linear(self.num_rows, self.num_rows)
        self.LinearCol2 = nn.Linear(self.num_rows, self.num_actions)

    def forward(self, x):
        # for idx, sample in enumerate(x):
        #     sample = (sample - torch.mean(sample, dim=0)) / (self.eps + torch.std(sample, dim=0))
        #     x[idx] = sample
        # import pdb; pdb.set_trace()
        # print(x.dtype, x.shape)
        # x = self.LinearRow2(x)
        # print(x.dtype, x.shape)
        # x = self.relu(x)
        # x = x.view(-1, self.num_rows)
        # print(x.dtype, x.shape)

        # x = self.LinearCol2(x)
        # print(x.dtype, x.shape)

        # x = self.bn_row(x)
        x = x[:, :, -1]
        x[:, 0] = 1
        x[:, 1:] = 0
        # import pdb; pdb.set_trace()
        x = self.LinearCol2(x)
        # x = self.relu(x)
        # x = self.LinearCol2(x)

        return x
