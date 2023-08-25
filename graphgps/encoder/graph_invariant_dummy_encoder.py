import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
import networkx as nx
import numpy as np
import torch.nn.functional as F


@register_node_encoder('GIDummyEncoder')
class GraphInvariantMaskEncoder(torch.nn.Module):
    def __init__(self, dim_emb=None):
        super().__init__()


    def forward(self, batch):
        return batch
