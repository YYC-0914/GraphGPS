"""
Code taken from ogb examples and adapted
"""

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import GINConv as PyGINConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from torch_sparse import matmul
from torch import Tensor
# import pdb; pdb.set_trace()

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GraphConvE(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super().__init__(aggr="add")

        self.in_channels = in_dim
        self.out_channels = emb_dim

        if isinstance(self.in_channels, int):
            self.in_channels = (self.in_channels, self.in_channels)

        self.lin_rel = torch.nn.Linear(self.in_channels[0], self.out_channels)
        self.lin_root = torch.nn.Linear(self.in_channels[1], self.out_channels, bias=False)
        self.bond_encoder = BondEncoder(emb_dim=self.out_channels)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_embedding)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_root(x_r)

        return out

    def message(self, x_j, edge_weight):
        return self.lin_rel(x_j) if edge_weight is None else self.lin_rel(x_j) + edge_weight


class ZINCGraphConvE(GraphConvE):
    def __init__(self, in_dim, emb_dim):
        super(ZINCGraphConvE, self).__init__(in_dim, emb_dim)
        self.bond_encoder = torch.nn.Embedding(4, emb_dim)


class ZINCGINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(ZINCGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = torch.nn.Embedding(4, in_dim)

    def forward(self, x, edge_index, edge_attr):
        # breakpoint()
        # print("edge_attr: ", edge_attr)
        # print("edge_attrr: ", edge_attr.dtype)
        # edge_attr = edge_attr.long()
        # print("edge_attr: ", edge_attr)
        # print("edge_attrr: ", edge_attr.dtype)
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class OriginalGINConv(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(OriginalGINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        self.layer = PyGINConv(nn=mlp, train_eps=False)

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(in_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + \
               F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
