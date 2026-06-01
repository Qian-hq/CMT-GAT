"""
CMT-GAT Model Architecture

This module implements the core CMT-GAT (Contrastive Multi-Head Trend-Aware
Graph Attention) model, which consists of three key components:

1. Temporal Trend-Aware Multi-Head Attention (TTA-MHA):
   Captures temporal trend information through causal convolution-based
   query/key generation before computing scaled dot-product attention.

2. Contrastive Learning Branch:
   Projects masked and original trajectory representations into an embedding
   space and regularizes them with MSE contrastive loss, encouraging
   consistent encodings under missing data conditions.

3. Graph Attention Network (GAT):
   Models spatial relationships among surrounding vehicles and environmental
   context to produce final lane-change intention predictions.

Reference:
    "CMT-GAT: A Contrastive Learning Framework for Predicting Lane Change
    Manoeuvres of Surrounding Vehicles in Weaving Areas under Missing Data Conditions"
"""

from torch import nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import torch
from torch_geometric.nn import GATConv


class TrendAwareAttention(nn.Module):
    """
    Temporal Trend-Aware Multi-Head Attention (TTA-MHA).

    Unlike standard multi-head attention that uses linear projections for Q/K/V,
    this module generates Query and Key via 1D causal convolutions, enabling each
    attention head to capture local temporal trends before computing global dependencies.
    Value is obtained via a standard linear projection.

    The causal convolution (kernel_size > 1 with appropriate padding + truncation)
    ensures that attention at time t only depends on t' <= t, preserving causality.

    Args:
        K:           Number of attention heads.
        d:           Dimension per attention head. Total output dim = K * d.
        kernel_size: Convolution kernel size for Q/K generation.

    Input:
        X: (batch_size, num_steps, num_vertices, D) where D = K * d

    Output:
        (batch_size, num_steps, num_vertices, D)
    """

    def __init__(self, K, d, kernel_size):
        super(TrendAwareAttention, self).__init__()
        self.D = K * d
        self.d = d
        self.K = K
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        # Causal 2D conv layers for trend-aware Q and K generation
        # Operate on (B, D, N, T): kernel=(1, k) along time axis only
        self.conv_q = nn.Conv2d(self.D, self.D, (1, kernel_size),
                                padding=(0, self.padding))
        self.conv_k = nn.Conv2d(self.D, self.D, (1, kernel_size),
                                padding=(0, self.padding))

        # Batch normalization for stable training after convolutions
        self.norm_q = nn.BatchNorm2d(self.D)
        self.norm_k = nn.BatchNorm2d(self.D)

        # Linear projection for value (no trend extraction needed)
        self.fc_v = nn.Linear(self.D, self.D)

        # Output fusion layer to combine multi-head outputs
        self.fc_out = nn.Linear(self.D, self.D)

    def forward(self, X):
        """
        Forward pass of TTA-MHA.

        Steps:
          1. Permute input from (B,T,N,D) to (B,D,N,T) for 2D conv.
          2. Apply causal convolutions -> trend-aware Q, K.
          3. Truncate padded positions to maintain original sequence length.
          4. Split into multi-heads, compute scaled dot-product attention.
          5. Concatenate heads and fuse through output projection.
        """
        batch_size = X.shape[0]

        # Reshape for 2D conv: (B, T, N, D) --> (B, D, N, T)
        X_ = X.permute(0, 3, 2, 1)

        # Generate Q, K via causal convolutions; V via linear projection
        # After conv + norm + permute + truncate: back to (B, T, N, D)
        query = self.norm_q(self.conv_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)
        key = self.norm_k(self.conv_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)
        value = self.fc_v(X)

        # Multi-head split: (B, T, N, D) --> (B*K, T, N, d)
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        # Reshape for batched matmul
        query = query.permute(0, 2, 1, 3)   # (B*K, N, T, d)
        key = key.permute(0, 2, 3, 1)       # (B*K, N, d, T)
        value = value.permute(0, 2, 1, 3)    # (B*K, N, T, d)

        # Scaled dot-product attention with temperature scaling
        attn_weights = (query @ key) * (self.d ** -0.5)  # (B*K, N, T, T)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted aggregation of values
        out = attn_weights @ value  # (B*K, N, T, d)

        # Merge heads: (B*K, N, T, d) --> (B, N, T, D)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)

        # Output projection for feature fusion across subspaces
        out = self.fc_out(out)

        return out.permute(0, 2, 1, 3)  # (B, T, N, D)


class SpatialGAT(nn.Module):
    """
    Graph Attention Network module for spatial context modeling.

    Takes encoded trajectory features of surrounding vehicles along with
    environmental context and edge attributes, applies GAT convolution,
    then classifies lane-change intentions via an MLP head.

    The fixed edge index defines a star topology: node 0 (ego vehicle)
    connects bidirectionally to nodes 1-6 (surrounding vehicles).

    Args:
        in_features: Dimension of input node features (encoded trajectory dim).
        hidden_size: Hidden dimension of GAT convolution output per node.
    """

    def __init__(self, in_features, hidden_size):
        super(SpatialGAT, self).__init__()

        # Single-layer GAT convolution (1 head, concat mode)
        self.gat_conv = GATConv(in_features, hidden_size, heads=1, concat=True)

        # Classifier: concatenates GAT output (7 nodes x hidden_dim) with env features
        gat_output_dim = 7 * hidden_size  # ego + 6 surrounding vehicles
        env_feature_dim = 227 - gat_output_dim  # remaining env dimensions
        self.classifier = nn.Linear(gat_output_dim + env_feature_dim, 30)

    def forward(self, x_encoded, batch_env):
        """
        Forward pass through the spatial GAT branch.

        Args:
            x_encoded: Encoded trajectory features, shape (B, T, N, hidden).
                       Last timestep encoding is used as vehicle representation.
            batch_env: Environmental context, shape (B, 227). Contains edge weights
                       (first 6 dims) and scene-level features (remaining dims).

        Returns:
            logits: Class prediction logits, shape (B, num_classes=30).
        """
        B = x_encoded.shape[0]

        # Fixed star-graph edge index: ego <-> neighbors
        edge_index = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6]
        ], device=x_encoded.device)

        # Edge attributes from environment data (relative distance/velocity)
        edge_attr = batch_env[:, :6]

        # Apply GAT per sample (graph structure is identical but edge attrs vary)
        gat_outputs = []
        for i in range(B):
            sample_enc = x_encoded[i]      # (T, N, hidden)
            sample_edge = edge_attr[i]     # (6,)
            gat_out = self.gat_conv(sample_enc, edge_index, sample_edge)  # (T, N, hidden)
            gat_outputs.append(gat_out[-1])  # Use final timestep: (N, hidden)

        # Stack: (B, N*hidden)
        x_gat = torch.stack(gat_outputs, dim=0).view(B, -1)

        # Concatenate with environmental context
        env_context = batch_env[:, 6:]
        combined = torch.cat([x_gat, env_context], dim=1)

        return self.classifier(combined)


class Network(nn.Module):
    """
    Complete CMT-GAT network combining all three modules:

    Architecture pipeline:
        Masked Trajectory  --> FC Embedding --> TTA-MHA --> Projector --> z_i
                                                              |-> MSE Loss
        Original Trajectory --> FC Embedding --> TTA-MHA --> Projector --> z_j

        Masked Encoding --> Spatial GAT(env) --> MLP --> Lane-change Prediction

    The model processes paired inputs (masked + original trajectories) through shared
    TTA-MHA encoders, producing contrastive embeddings for regularization and
    spatial-GAT-based predictions for classification.

    Args:
        dropout_rate: Dropout probability. Default 0.1 (as reported in experiments).
    """

    def __init__(self, dropout_rate=0.1):
        super(Network, self).__init__()

        # ---- Hyperparameters (Table 6, manuscript) ----
        self.num_heads = 4              # K: number of attention heads
        self.head_dim = 8               # d: dimension per head (D = K*d = 32)
        self.kernel_size = 3            # Conv kernel size for Q/K generation
        self.input_dim = 19             # Raw feature dim per vertex per timestep
        self.embed_dim = 32             # FC output dim (= K * d = 32)
        self.num_classes = 30           # Lane-change intention categories

        # ---- Feature Embedding Layer ----
        self.feature_embed = nn.Linear(self.input_dim, self.embed_dim)

        # ---- TTA-MHA Encoder (shared weights for both branches) ----
        self.tta_attention = TrendAwareAttention(
            K=self.num_heads,
            d=self.head_dim,
            kernel_size=self.kernel_size
        )

        # ---- Regularization ----
        self.dropout = nn.Dropout(dropout_rate)

        # ---- Contrastive Projection Head (MLP) ----
        # Projects flattened TTA-MHA output to embedding space
        self.instance_projector = nn.Sequential(
            nn.Linear(self.embed_dim * 7 * 10, 128),  # 7 vertices x 10 timesteps x embed_dim
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.num_classes),
        )

        # ---- Spatial GAT Classifier ----
        self.spatial_gat = SpatialGAT(
            in_features=self.embed_dim,
            hidden_size=32
        )

    def forward(self, batch_X_masked, batch_X_original, batch_env):
        """
        Full forward pass of the CMT-GAT model.

        Args:
            batch_X_masked:   Missing-data-augmented trajectory tensor (B, T, N, D_in).
            batch_X_original: Complete (unmasked) trajectory tensor (B, T, N, D_in).
            batch_env:        Environmental context tensor (B, env_dim).

        Returns:
            z_i:    L2-normalized embedding of masked trajectory representation.
            z_j:    L2-normalized embedding of original trajectory representation.
            output: Classification logits (B, num_classes).
        """
        B, T, N, D_in = batch_X_masked.size()

        # Encode both trajectory branches through shared TTA-MHA encoder
        h_masked = self._encode_trajectory(batch_X_masked)
        h_orig = self._encode_trajectory(batch_X_original)

        # --- Contrastive projection ---
        h_flat_m = h_masked.permute(0, 2, 1, 3).reshape(B * N, -1)
        h_flat_o = h_orig.permute(0, 2, 1, 3).reshape(B * N, -1)

        z_i = normalize(self.instance_projector(h_flat_m), dim=1)
        z_j = normalize(self.instance_projector(h_flat_o), dim=1)

        # --- Spatial graph classification (using masked encoding) ---
        output = self.spatial_gat(h_masked, batch_env)

        return z_i, z_j, output

    def _encode_trajectory(self, traj_tensor):
        """
        Encode raw trajectory through linear projection + TTA-MHA.

        Args:
            traj_tensor: Raw input of shape (B, T, N, D_in).

        Returns:
            Encoded representation of shape (B, T, N, embed_dim).
        """
        B, T, N, D_in = traj_tensor.size()
        # Per-vertex linear embedding
        h = self.feature_embed(traj_tensor.reshape(-1, D_in))  # (B*T*N, embed_dim)
        h = h.reshape(B, T, N, -1)                            # (B, T, N, embed_dim)
        # Temporal trend-aware multi-head attention
        h = self.tta_attention(h)
        return h
