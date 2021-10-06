import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, n_layers, hid_dim, out_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GINEConv(GraphEncoder.MLP(num_node_features, hid_dim)))
        for _ in range(1, n_layers):
            self.convs.append(GINEConv(GraphEncoder.MLP(hid_dim, hid_dim)))
        self.out_lin = Linear(hid_dim, out_dim)
        self.edge_lins = ModuleList()
        self.edge_lins.append(nn.Linear(num_edge_features, num_node_features))
        for _ in range(1, n_layers):
            self.edge_lins.append(nn.Linear(num_edge_features, hid_dim))
        self.act = nn.ReLU(inplace=True)
        self.n_layers = n_layers
        self.dropout = dropout

    @staticmethod
    def MLP(in_channels, out_channels):
        return Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.n_layers):
            edge_attr_transform = self.edge_lins[i](edge_attr)
            x = self.convs[i](x, edge_index, edge_attr_transform)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.act(x)
        x = self.out_lin(x)
        x = scatter(x, data.batch, dim=0, reduce="sum")
        return x


class GraphDecoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim, emb_dim, cond_dim, n_rnn_layers,
                 rnn_hid_dim, out_dim, dropout, device):
        super(GraphDecoder, self).__init__()
        self.fragment_embed = GraphEncoder(num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim, emb_dim,
                                           dropout)
        self.rnn = nn.GRU(emb_dim + cond_dim, rnn_hid_dim, n_rnn_layers, dropout=dropout, batch_first=True)
        self.cond_2_hid = nn.Linear(cond_dim, rnn_hid_dim)
        self.decode = nn.Linear(rnn_hid_dim, out_dim)
        self.act = nn.ReLU(inplace=True)
        self.n_rnn_layers = n_rnn_layers
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device

    def forward(self, graph, label, length, cond):
        n_batch, n_fragment = label.shape
        cnt = 0
        # Calculate h0
        h_0 = self.cond_2_hid(cond)
        h_0 = h_0.unsqueeze(0).repeat(self.n_rnn_layers, 1, 1)
        # Reshape condition
        cond = cond.unsqueeze(1)
        cond = cond.repeat(1, n_fragment, 1)
        # Calculate fragment embedding
        frag_embedding = self.fragment_embed(graph)
        # Reshape fragment embedding
        frag_embedding_padded = torch.zeros(n_batch, n_fragment+1, self.emb_dim, dtype=torch.float32).to(self.device)
        for i in range(n_batch):
            frag_embedding_padded[i, 1:(1 + length[i]), :] = frag_embedding[cnt:(cnt + length[i])]
            cnt += length[i]
        frag_embedding = frag_embedding_padded[:, :-1, :]
        frag_embedding = torch.cat([frag_embedding, cond], dim=-1)
        # Prepare input for rnn
        frag_embedding = nn.utils.rnn.pack_padded_sequence(frag_embedding, length.cpu(), batch_first=True,
                                                           enforce_sorted=False)
        output, _ = self.rnn(frag_embedding, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=n_fragment)
        output = self.decode(output)
        return output


class FAME(nn.Module):
    def __init__(self, num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim, latent_dim, emb_dim, out_dim,
                 n_rnn_layers, rnn_hid_dim, dropout, device):
        super(FAME, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim,
                                          latent_dim * 2, dropout)
        self.graph_decoder = GraphDecoder(num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim, emb_dim,
                                          latent_dim, n_rnn_layers, rnn_hid_dim, out_dim, dropout, device)
        self.latent_dim = latent_dim
        self.device = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, graph_drug, graph_frag, batch_frag, label):
        mu, log_var = torch.split(self.graph_encoder(graph_drug), self.latent_dim, -1)
        latent = self.reparameterize(mu, log_var)
        output = self. graph_decoder(graph_frag, label, batch_frag, latent)
        return output, mu, log_var

    def loss(self, recon_seq, label_seq, length, mu, logvar, reduce=True):
        recon_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none') \
            (recon_seq.reshape(-1, recon_seq.shape[-1]), label_seq.reshape(-1))
        recon_loss = (recon_loss.reshape(label_seq.shape[0], -1).sum(dim=-1) / length.to(self.device))
        if reduce:
            recon_loss = recon_loss.mean()
        kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return recon_loss, kld_loss
