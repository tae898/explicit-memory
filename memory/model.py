import logging
import os
import torch

from torch import nn

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self, num_rows: int, num_cols: int, num_actions: int):
        """
        Args
        ----
        num_rows: number of rows
        num_cols: number of columns
        num_actions: number of discrete actions available in the environment

        """
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.bn_row = nn.BatchNorm1d(num_cols)
        self.bn_col = nn.BatchNorm1d(num_rows)
        self.dropout = nn.Dropout(0.5)

        self.LinearRow1 = nn.Linear(num_cols, num_cols)
        self.LinearRow2 = nn.Linear(num_cols, 1)

        self.LinearCol1 = nn.Linear(num_rows, num_rows)
        self.LinearCol2 = nn.Linear(num_rows, num_actions)

    def forward(self, x):
        # # x = x.float()
        # x = self.LinearRow1(x)
        # x = self.relu(x)
        # # x = self.bn_row(x)
        # x = self.dropout(x)

        # x = self.LinearRow1(x)
        # x = self.relu(x)
        # # x = self.bn_row(x)
        # x = self.dropout(x)

        # x = self.LinearRow2(x)
        # x = self.relu(x)

        # x = x.view(-1, self.num_rows)

        # x = self.LinearCol1(x)
        # x = self.relu(x)
        # # x = self.bn_col(x)
        # x = self.dropout(x)

        # x = self.LinearCol1(x)
        # x = self.relu(x)
        # # x = self.bn_col(x)
        # x = self.dropout(x)

        # x = self.LinearCol2(x)

        # import pdb

        # pdb.set_trace()
        x[..., -1] = torch.arange(1, self.num_rows+1)
        x = x[..., -1]
        # print(x.round())
        x = self.LinearCol2(x)
        # x = self.relu(x)
        # x[..., 1:] = 0
        # x[..., 0] = 1

        return x
