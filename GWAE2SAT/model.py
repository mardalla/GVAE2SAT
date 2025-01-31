import torch
import torch.nn.functional as F
# from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.models.autoencoder import VGAE

class AttentionAggregation(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = torch.nn.Linear(in_channels, 1)
    
    def forward(self, x):
        scores = self.attention(x).squeeze(-1)  # [num_nodes]
        alpha = torch.softmax(scores, dim=0)
        return (x * alpha.unsqueeze(-1)).sum(dim=0)

class VGAEInference(torch.nn.Module):
    def __init__(self, in_shape, out_size, layer_size, gnn_layers, device):
        super().__init__()
        self.device = device
        
        # Initialize GAT layers
        self.gat_layers = torch.nn.ModuleList()
        input_dim = in_shape[1]
        for _ in range(gnn_layers):
            self.gat_layers.append(
                GATConv(input_dim, layer_size, add_self_loops=False, device=device)
            )
            input_dim = layer_size  # Output dimension becomes input for next layer
        
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(layer_size, device=device) for _ in range(gnn_layers)
        ])
        
        # Attention-based aggregation
        self.pooling = AttentionAggregation(layer_size)
        
        self.mu_dense = torch.nn.Linear(layer_size, out_size)
        self.std_dense = torch.nn.Linear(layer_size, out_size)

    def forward(self, adj):
        # Convert dense adjacency matrix to edge_index
        edge_index = adj.nonzero().t().contiguous()
        
        # Compute degree-normalized adjacency (if needed)
        x = torch.eye(adj.size(0), device=self.device)  # Identity features as input
        
        # Process through GAT layers
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            x = F.selu(x)
            x = self.layer_norms[i](x)
        
        # Aggregate nodes with attention
        x = self.pooling(x)
        
        mu = self.mu_dense(x)
        std = self.std_dense(x)
        return mu, std

class DenseDecoder(torch.nn.Module):
    def __init__(self, in_size, out_shape, layer_size, expansion_layers, dropout, device):
        super().__init__()
        self.out_shape = out_shape
        self.dropout = dropout
        
        # Add self-attention layer in decoder
        self.self_attention = torch.nn.MultiheadAttention(layer_size, num_heads=1, device=device)
        
        self.expansions = torch.nn.ModuleList([
            torch.nn.Linear(in_size, layer_size, device=device),
            *[torch.nn.Linear(layer_size, layer_size, device=device) 
              for _ in range(expansion_layers-2)],
            torch.nn.Linear(layer_size, out_shape[0]*out_shape[1], device=device)
        ])
        
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(layer_size, device=device) 
            for _ in range(expansion_layers-1)
        ])

    def forward(self, z):
        x = F.selu(self.expansions[0](z))
        for i in range(1, len(self.expansions)):
            # Apply self-attention
            x_attn, _ = self.self_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = x + x_attn.squeeze(0)
            
            x = F.dropout(x, self.dropout)
            x = self.layer_norms[i-1](x)
            x = F.selu(self.expansions[i](x))
        
        adj = torch.tanh(x.reshape(self.out_shape))
        return adj

def get_vgae(max_shape, latent_size, device,
             enc_layer_size, dec_layer_size,
             num_gnn, num_expansions, dropout):
    vgae = VGAE(VGAEInference(max_shape, latent_size,
                              enc_layer_size, num_gnn, device),
                DenseDecoder(latent_size, max_shape,
                             dec_layer_size, num_expansions,
                             dropout, device))
    vgae.to(device)
    return vgae