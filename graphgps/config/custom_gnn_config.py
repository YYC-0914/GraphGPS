from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.sublayer_residual = False
    cfg.gnn.num_clusters = 3
    cfg.gnn.train_eps = False
    cfg.gnn.scaling = 3
    cfg.gnn.GIMaskEncoder_layers = 2 
    cfg.gnn.GIMaskEncoder_raw_norm_type = None
    cfg.gnn.GIMaskEncoder_hidden_dim = 8
    cfg.gnn.GIMaskEncoder_graph_invariant = ["degree", "pagerank"]
    cfg.gnn.GIMaskEncoder_batch_norm = True 
    cfg.gnn.HeatConvBlock_n_layers = 1
    cfg.gnn.layers_block = 5
    cfg.gnn.subgraph_encoder = CN()
    cfg.gnn.subgraph_encoder.emb_dim = 64
    cfg.gnn.subgraph_encoder.use_transpose = False 
    cfg.gnn.subgraph_encoder.dropout = 0.0
    cfg.gnn.subgraph_encoder.residual = False
    cfg.gnn.subgraph_encoder.batchnorm = True
    cfg.gnn.subgraph_encoder.use_readout = True
    cfg.gnn.subgraph_encoder.use_mlp = True
    cfg.gnn.subgraph_encoder.subgraph_readout = "sum"
    cfg.gnn.subgraph_encoder.feature_encoder = "zinc"
    cfg.gnn.subgraph_encoder.gnn_type = "zincgin"
    cfg.gnn.subgraph_encoder.num_layers = 4
