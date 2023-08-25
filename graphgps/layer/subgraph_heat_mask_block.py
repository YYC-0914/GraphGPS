import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg
from graphgps.layer.heat_mask_layer import HeatConvLayer
from graphgps.layer.subgraph_mask_encoder import SUNMaskEncoder

from graphgps.layer.subgraph_conv_layer import GINConv
from graphgps.layer.subgraph_conv_layer import OriginalGINConv
from graphgps.layer.subgraph_conv_layer import GraphConvE
from graphgps.layer.subgraph_conv_layer import GCNConv
from graphgps.layer.subgraph_conv_layer import ZINCGINConv
from graphgps.layer.subgraph_conv_layer import ZINCGraphConvE
from graphgps.layer.subgraph_conv_layer import GraphConv
from graphgps.layer.subgraph_mask_encoder import ZincAtomEncoder



class SubgraphHeatConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers):
        super().__init__()
        emb_dim = cfg.gnn.subgraph_encoder.emb_dim
        policy = cfg.dataset.subgraph_policy
        # setting up feature encoder for subgraph encoder
        feature_encoder = cfg.gnn.subgraph_encoder.feature_encoder
        if feature_encoder == "zinc":
            encoder = ZincAtomEncoder(policy=policy, emb_dim=emb_dim)
        # setting up GNNConv layer for subgraph encoder    
        gnn_type = cfg.gnn.subgraph_encoder.gnn_type
        if gnn_type == 'gin':
            GNNConv = GINConv
        elif gnn_type == 'originalgin':
            GNNConv = OriginalGINConv
        elif gnn_type == 'graphconv':
            GNNConv = GraphConv
        elif gnn_type == 'graphconvedge':
            GNNConv = GraphConvE
        elif gnn_type == 'zincgraphconvedge':
            GNNConv = ZINCGraphConvE
        elif gnn_type == 'gcn':
            GNNConv = GCNConv
        elif gnn_type == 'zincgin':
            GNNConv = ZINCGINConv
        else:
            raise ValueError('Undefined GNN type called {}'.format(gnn_type))

        in_dim = emb_dim if policy != "ego_nets_plus" else emb_dim + 2

        self.mask_encoder = SUNMaskEncoder(num_layers=cfg.gnn.subgraph_encoder.num_layers,
                                           in_dim=in_dim,
                                           emb_dim=emb_dim,
                                           num_clusters=cfg.gnn.num_clusters,
                                           feature_encoder=encoder,
                                           GNNConv=GNNConv,
                                           use_transpose=cfg.gnn.subgraph_encoder.use_transpose,
                                           drop_ratio=cfg.gnn.subgraph_encoder.dropout,
                                           res=cfg.gnn.subgraph_encoder.residual,
                                           add_bn=cfg.gnn.subgraph_encoder.batchnorm,
                                           use_readout=cfg.gnn.subgraph_encoder.use_readout,
                                           use_mlp=cfg.gnn.subgraph_encoder.use_mlp,
                                           subgraph_readout=cfg.gnn.subgraph_encoder.subgraph_readout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        for layer in range(n_layers):
            if layer == 0:
                self.layers.append(HeatConvLayer(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    dropout=cfg.gnn.dropout,
                    train_eps=cfg.gnn.train_eps,
                    batch_norm=cfg.gnn.batchnorm,
                    sublayer_residual=cfg.gnn.sublayer_residual,
                    residual=cfg.gnn.residual,
                    num_clusters=cfg.gnn.num_clusters
                ))
            else:
                self.layers.append(HeatConvLayer(
                    dim_in=dim_out,
                    dim_out=dim_out,
                    dropout=cfg.gnn.dropout,
                    train_eps=cfg.gnn.train_eps,
                    batch_norm=cfg.gnn.batchnorm,
                    sublayer_residual=cfg.gnn.sublayer_residual,
                    residual=cfg.gnn.residual,
                    num_clusters=cfg.gnn.num_clusters
                ))
            self.bns.append(nn.BatchNorm1d(dim_out))

    def forward(self, batch, cur_layer):
        self.mask_encoder(batch, cur_layer)
        for i in range(self.n_layers):
            self.layers[i](batch, cur_layer)
        return batch
