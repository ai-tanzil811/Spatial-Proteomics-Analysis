"""
05_graphsage_model.py
GraphSAGE model for sample-level survival risk prediction.

Architecture:
  - Interaction-type embedding added to each source node feature vector
  - 3× SAGEConv layers (mean aggregation) with distance-weighted messages
  - BatchNorm + ReLU + Dropout after each conv
  - Global pooling: concat(mean_pool, max_pool) → 128-dim
  - Risk head: Linear(128→64) → ReLU → Dropout → Linear(64→1) → scalar θ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops


class DistanceWeightedSAGEConv(MessagePassing):
    """
    SAGEConv (mean aggregation) with distance-weighted message passing.

    Message from j→i is weighted by edge_attr[:, 1] (distance_weight).
    The interaction-type embedding is NOT handled here; it is added to
    node features before the first layer.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")   # we normalise by sum-of-weights ourselves
        self.lin_self  = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x         : (N, in_channels)
        edge_index: (2, E)
        edge_attr : (E, 3)  — col 1 is distance_weight ∈ [0,1]
        """
        dist_weight = edge_attr[:, 1].unsqueeze(1)   # (E, 1)
        agg = self.propagate(edge_index, x=x, dist_weight=dist_weight)
        out = self.lin_self(x) + self.lin_neigh(agg) + self.bias
        return out

    def message(self, x_j: torch.Tensor, dist_weight: torch.Tensor) -> torch.Tensor:
        return dist_weight * x_j

    def aggregate(self, inputs: torch.Tensor,
                  index: torch.Tensor,
                  ptr=None,
                  dim_size=None) -> torch.Tensor:
        # Weighted mean: sum(w*x) / count  — native PyTorch, no torch-scatter required
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        weighted_sum = torch.zeros(dim_size, inputs.size(1),
                                   dtype=inputs.dtype, device=inputs.device)
        weighted_sum.scatter_add_(0, index.unsqueeze(1).expand_as(inputs), inputs)
        count = torch.zeros(dim_size, 1, dtype=inputs.dtype, device=inputs.device)
        count.scatter_add_(0, index.unsqueeze(1),
                           torch.ones(inputs.size(0), 1,
                                      dtype=inputs.dtype, device=inputs.device))
        return weighted_sum / count.clamp(min=1e-6)


class GraphSAGESurvival(nn.Module):
    """
    3-layer GraphSAGE + mean/max global pooling + Cox risk head.

    Parameters
    ----------
    in_channels : number of node features (45 by default)
    hidden_dim  : hidden channel size (64 by default)
    n_layers    : number of SAGEConv layers (3 by default)
    dropout     : dropout probability
    n_interaction_types : number of interaction type categories
    interaction_embed_dim : embedding dimension for interaction type
    """

    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 dropout: float = 0.3,
                 n_interaction_types: int = 5,
                 interaction_embed_dim: int = 8):
        super().__init__()

        self.dropout = dropout

        # Interaction type embedding (added to source node before each conv)
        self.interaction_embed = nn.Embedding(n_interaction_types,
                                              interaction_embed_dim)

        # First layer: in_channels + interaction_embed_dim → hidden_dim
        # Subsequent layers: hidden_dim + interaction_embed_dim → hidden_dim
        first_in = in_channels + interaction_embed_dim
        rest_in  = hidden_dim  + interaction_embed_dim

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(DistanceWeightedSAGEConv(first_in, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_layers - 1):
            self.convs.append(DistanceWeightedSAGEConv(rest_in, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Risk head
        pool_dim = hidden_dim * 2   # concat mean + max
        self.risk_head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        """
        Returns risk score tensor of shape (batch_size, 1).
        """
        x          = data.x           # (N_total, in_channels)
        edge_index = data.edge_index  # (2, E_total)
        edge_attr  = data.edge_attr   # (E_total, 3)
        batch      = data.batch       # (N_total,)

        # Interaction type: integer in edge_attr[:, 2]
        itype = edge_attr[:, 2].long().clamp(0, self.interaction_embed.num_embeddings - 1)
        itype_emb = self.interaction_embed(itype)   # (E, embed_dim)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Add interaction embedding to source node features via edge
            x_src = x[edge_index[0]]  # (E, in_channels)
            # Broadcast: attach embedding to edge_src; we materialise a new x
            # by average-pooling interaction embeddings incident on each node
            # (simple, avoids variable-dim aggregation)
            itype_node = torch.zeros(x.size(0), itype_emb.size(1),
                                     device=x.device)
            itype_node.scatter_add_(0,
                                    edge_index[0].unsqueeze(1).expand_as(itype_emb),
                                    itype_emb)
            degree = torch.zeros(x.size(0), 1, device=x.device)
            degree.scatter_add_(0, edge_index[0].unsqueeze(1),
                                 torch.ones(edge_index.size(1), 1, device=x.device))
            itype_node = itype_node / degree.clamp(min=1)

            x_aug = torch.cat([x, itype_node], dim=1)   # (N, in + embed_dim)
            x = conv(x_aug, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        mean_pool = global_mean_pool(x, batch)   # (B, hidden_dim)
        max_pool  = global_max_pool(x,  batch)   # (B, hidden_dim)
        graph_emb = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2*hidden_dim)

        risk = self.risk_head(graph_emb)   # (B, 1)
        return risk


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from config import N_NODE_FEATURES, HIDDEN_DIM, N_LAYERS, DROPOUT, N_INTERACTION_TYPES, INTERACTION_EMBED_DIM

    model = GraphSAGESurvival(
        in_channels          = N_NODE_FEATURES,
        hidden_dim           = HIDDEN_DIM,
        n_layers             = N_LAYERS,
        dropout              = DROPOUT,
        n_interaction_types  = N_INTERACTION_TYPES,
        interaction_embed_dim = INTERACTION_EMBED_DIM,
    )
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
