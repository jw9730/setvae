import torch.nn as nn
import torch.nn.functional as F
from .ops import masked_fill, MaskedBatchNorm1d


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_hidden, activation='relu', residual=False, use_bn=False, use_masked_bn=False, dropout_p=0.):
        super().__init__()
        assert activation in ['relu', 'tanh']
        self.activation = activation
        self.residual = residual
        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(n_hidden):
            self.fc.append(nn.Linear(dim_in if i == 0 else dim_hidden, dim_hidden))
            self.bn.append((MaskedBatchNorm1d(dim_hidden) if use_masked_bn else nn.BatchNorm1d(dim_hidden)) if use_bn
                           else None)
        self.out = nn.Linear(dim_hidden if n_hidden > 0 else dim_in, dim_out)

    def forward(self, x):  # [B, C]
        x_input = x.clone()
        for fc, bn in zip(self.fc, self.bn):  # FC - BN - ReLU - FC
            x = fc(x)
            if bn is not None:
                x = bn(x)
            x = F.relu(x) if self.activation == 'relu' else x.tanh()
            x = self.dropout(x) if hasattr(self, 'dropout') else x
        if self.residual:
            x = x.clone() + x_input
        x = self.out(x)
        return x


class ElementwiseMLP(MLP):
    def __init__(self, dim_in, dim_hidden, dim_out, n_hidden, activation='relu', residual=False, use_bn=False, use_masked_bn=False, dropout_p=0.):
        super().__init__(dim_in, dim_hidden, dim_out, n_hidden, activation, residual, use_bn, use_masked_bn, dropout_p)

    def forward(self, x, x_mask=None):  # [B, N, C]
        x = masked_fill(x, x_mask)
        x_input = x.clone()
        for fc, bn in zip(self.fc, self.bn):  # FC - BN - ReLU - FC
            x = fc(x)
            x = masked_fill(x, x_mask)  # commenting this line should make no change
            x = x.transpose(1, 2)
            if bn is not None:
                x = bn(x) if x_mask is None else bn(x, x_mask)  # BatchNorm1D takes [B, C, N]-shaped input
            x = x.transpose(1, 2)
            x = F.relu(x) if self.activation == 'relu' else x.tanh()
            x = self.dropout(x) if hasattr(self, 'dropout') else x
        if self.residual:
            x = x.clone() + x_input
        x = masked_fill(x, x_mask)
        x = self.out(x)
        return x
