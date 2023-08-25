import argparse
import logging
import math
import os
import os.path as osp
from posixpath import split
import random
import shutil
from typing import Optional, Union, Tuple
from pathlib import Path
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import TUDataset as TUDataset_
from torch_geometric.datasets import ZINC
from torch_geometric.transforms import OneHotDegree, Constant
from torch_geometric.utils import k_hop_subgraph



ORIG_EDGE_INDEX_KEY = 'original_edge_index'

class SubgraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == ORIG_EDGE_INDEX_KEY:
            return self.num_nodes_per_subgraph
        else:
            return super().__inc__(key, value, *args, **kwargs)



class Graph2Subgraph:
    def __init__(self, process_subgraphs=lambda x: x, pbar=None):
        self.process_subgraphs = process_subgraphs
        self.pbar = pbar

    def __call__(self, data):
        assert data.is_undirected()
        subgraphs = self.to_subgraphs(data)
        subgraphs = [self.process_subgraphs(s) for s in subgraphs]

        batch = Batch.from_data_list(subgraphs)

        if self.pbar is not None: next(self.pbar)

        return SubgraphData(subgraph_x=batch.subgraph_x, subgraph_edge_index=batch.subgraph_edge_index, subgraph_edge_attr=batch.subgraph_edge_attr,
                            node2subgraph=batch.batch,
                            y=data.y, subgraph_idx=batch.subgraph_idx, subgraph_node_idx=batch.subgraph_node_idx,
                            num_subgraphs=len(subgraphs), num_nodes_per_subgraph=data.num_nodes, x = data.x,
                            edge_index=data.edge_index, edge_attr=data.edge_attr, original_edge_index=data.edge_index,
                            original_edge_attr=data.edge_attr)

    def to_subgraphs(self, data):
        raise NotImplementedError

class EgoNets(Graph2Subgraph):
    def __init__(self, num_hops, add_node_idx=False, process_subgraphs=lambda x: x, pbar=None):
        super().__init__(process_subgraphs, pbar)
        self.num_hops = num_hops
        self.add_node_idx = add_node_idx

    def to_subgraphs(self, data):

        subgraphs = []

        for i in range(data.num_nodes):

            _, _, _, edge_mask = k_hop_subgraph(i, self.num_hops, data.edge_index, relabel_nodes=False,
                                                num_nodes=data.num_nodes)
            subgraph_edge_index = data.edge_index[:, edge_mask]
            subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else data.edge_attr
            # breakpoint()
            # print("x feature: ", data.x)
            # print("x feature shape: ", data.x.shape)

            x = data.x
            if self.add_node_idx:
                # prepend a feature [0, 1] for all non-central nodes
                # a feature [1, 0] for the central node
                ids = torch.arange(2).repeat(data.num_nodes, 1)
                ids[i] = torch.tensor([ids[i, 1], ids[i, 0]])

                x = torch.hstack([ids, data.x]) if data.x is not None else ids.to(torch.float)

            subgraphs.append(
                Data(
                    subgraph_x=x, subgraph_edge_index=subgraph_edge_index, subgraph_edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs


def policy2transform(policy: str, num_hops, process_subgraphs=lambda x: x, pbar=None):
    if policy == "ego_nets":
        return EgoNets(num_hops, process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "ego_nets_plus":
        return EgoNets(num_hops, add_node_idx=True, process_subgraphs=process_subgraphs, pbar=pbar)
    else:
        raise ValueError("Invalid subgraph policy type")