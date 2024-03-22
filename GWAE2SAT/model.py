import torch
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.models.autoencoder import VGAE


class VGAEInference(torch.nn.Module):
    def __init__(self, in_shape, out_size, layer_size, gnn_layers, device):
        super().__init__()
        self.device = device
        self.ws = [torch.nn.Parameter(torch.rand([layer_size, layer_size],
                                                 device=device))
                   for g in range(gnn_layers - 1)]
        self.ws = [torch.nn.Parameter(torch.rand([in_shape[1],
                                                  layer_size],
                                                 device=device))] + self.ws

        self.layer_norms = [torch.nn.LayerNorm(layer_size, device=device)
                            for ln in range(gnn_layers+1)]
        self.pooling = MeanAggregation()

        self.mu_dense = torch.nn.Linear(layer_size, out_size)
        self.std_dense = torch.nn.Linear(layer_size, out_size)

    def forward(self, adj):        
        deg = abs(adj).sum(dim=1) * torch.eye(adj.size(0), device=self.device)
        deg.pow_(-1)
        deg.masked_fill_(deg == float("inf"), 0.)

        adj_hat = deg @ adj
        
        next = torch.nn.SELU()(adj_hat @ self.ws[0])
        next = self.layer_norms[0](next)
        for layer in range(1, len(self.ws)):
            if layer % 2 == 0:
                next = torch.nn.SELU()(adj_hat @ self.ws[layer])
            else:
                next = torch.nn.SELU()(adj_hat.T @ next @ self.ws[layer])
            next = self.layer_norms[layer](next)

        next = self.pooling(next)
        next = self.layer_norms[-1](next)

        mu = self.mu_dense(next)
        std = self.std_dense(next)

        return mu, std


class DenseDecoder(torch.nn.Module):
    def __init__(self, in_size, out_shape, layer_size, expansion_layers, dropout, device):
        super().__init__()
        self.out_shape = out_shape
        self.dropout = dropout
        self.expansions = [torch.nn.Linear(layer_size, layer_size, device=device)
                           for el in range(expansion_layers - 2)]
        self.expansions = [torch.nn.Linear(in_size, layer_size,
                                           device=device)] + self.expansions
        self.expansions = self.expansions + [torch.nn.Linear(layer_size,
                                                             out_shape[0]*out_shape[1],
                                                             device=device)]

        self.layer_norms = [torch.nn.LayerNorm(layer_size, device=device)
                            for ln in range(expansion_layers-1)]

    def forward(self, z):
        next = torch.nn.SELU()(self.expansions[0](z))
        for exp in range(1, len(self.expansions)):
            next = torch.nn.Dropout(self.dropout)(next)
            next = self.layer_norms[exp-1](next)
            next = torch.nn.SELU()(self.expansions[exp](next))

        adj = next.reshape(self.out_shape)
        return torch.tanh(adj)


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
