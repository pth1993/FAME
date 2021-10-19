from .model_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU
import numpy as np
import random
from torch_geometric.data import Data, Batch


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, n_layers, hid_dim, out_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GINEConv(GraphEncoder.MLP(num_node_features, hid_dim)))
        for _ in range(1, n_layers):
            self.convs.append(GINEConv(GraphEncoder.MLP(hid_dim, hid_dim)))
        # self.out_lin = Linear(hid_dim, out_dim)
        self.out_lin = Sequential(Linear(hid_dim, out_dim), nn.ReLU(inplace=True), Linear(out_dim, out_dim))
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
        x = scatter(x, data.batch, dim=0, reduce="mean")
        x = self.out_lin(x)
        return x


class GEEncoder(nn.Module):
    def __init__(self, out_dim, dropout=0.1, in_dim=64):
        super(GEEncoder, self).__init__()
        self.out_dim = out_dim
        self.ge_encoder = nn.Sequential(nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(),
                                        nn.Dropout(dropout), nn.Linear(64, out_dim))

    def forward(self, ge):
        return self.ge_encoder(ge)


class JointGraphEncoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, n_layers, hid_dim, out_dim, dropout):
        super(JointGraphEncoder, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, num_edge_features, n_layers, hid_dim, out_dim, dropout)
        self.ge_encoder = GEEncoder(out_dim)
        self.encoder = nn.Sequential(nn.Linear(out_dim * 2, out_dim * 2), nn.LeakyReLU(), nn.Linear(out_dim * 2, out_dim))

    def forward(self, graph, ge):
        z = torch.cat((self.graph_encoder(graph), self.ge_encoder(ge)), dim=-1)
        return self.encoder(z)


class JointGraphEncoderFT(nn.Module):
    def __init__(self, num_node_features, num_edge_features, n_layers, hid_dim, out_dim_pt, out_dim, dropout, checkpoint):
        super(JointGraphEncoderFT, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, num_edge_features, n_layers, hid_dim, out_dim_pt, dropout)
        self.graph_encoder.load_state_dict(checkpoint)
        for p in self.graph_encoder.parameters():
            p.requires_grad = False
        self.ft_layer = nn.Sequential(nn.Linear(out_dim_pt, 64), nn.BatchNorm1d(64), nn.LeakyReLU(),
                                      nn.Linear(64, out_dim))
        self.ge_encoder = GEEncoder(out_dim)
        self.encoder = nn.Sequential(nn.Linear(out_dim * 2, out_dim * 2), nn.LeakyReLU(), nn.Linear(out_dim * 2, out_dim))

    def forward(self, graph, ge):
        z = torch.cat((self.ft_layer(self.graph_encoder(graph)), self.ge_encoder(ge)), dim=-1)
        return self.encoder(z)


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
        self.out_dim = out_dim

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

    def sample(self, cond, neighbor, max_length, topk, atom_dict, bond_dict, fragment_dict):
        # cond = [batch * num_sample * cond_dim]
        n_batch, num_sample, cond_dim = cond.shape
        # output = torch.zeros(n_batch, num_sample, max_length, dtype=torch.int64).to(self.device)
        cond = cond.reshape(-1, cond_dim)
        # cond = [(batch * num_sample) * cond_dim]
        h = self.cond_2_hid(cond)
        # h = [(batch * num_sample) * rnn_hid_dim]
        h = h.unsqueeze(0).repeat(self.n_rnn_layers, 1, 1)
        # h = [n_layers * (batch * num_sample) * rnn_hid_dim]
        cond = cond.unsqueeze(1)
        # cond = [(batch * num_sample) * 1 * cond_dim]
        input = torch.zeros(n_batch * num_sample, 1, self.emb_dim, dtype=torch.float32).to(self.device)
        # input = [(batch * num_sample) * 1 * emb_dim]
        input = torch.cat([input, cond], dim=-1)
        # input = [(batch * num_sample) * 1 * (emb_dim + cond_dim)]
        fragment = []
        for i in range(max_length):
            o, h = self.rnn(input, h)
            # o = [(batch * num_sample) * 1 * rnn_hid_dim]
            # h = [n_layers * (batch * num_sample) * rnn_hid_dim]
            o = self.decode(o)
            # o = [(batch * num_sample) * 1 * out_dim]
            o = o.squeeze(1)
            # o = [(batch * num_sample) * out_dim]
            _, sorted_idx = torch.topk(o, self.out_dim - topk, dim=-1, largest=False)
            # sorted_idx = [(batch * num_sample) * topk]
            idx = np.tile(np.arange(n_batch * num_sample).reshape(n_batch * num_sample, 1), self.out_dim - topk)
            # idx = [(batch * num_sample) * topk]
            o[idx, sorted_idx] = -1e9
            # o = [(batch * num_sample) * out_dim]
            o = F.softmax(o, dim=-1)
            # o = [(batch * num_sample) * out_dim]
            p = torch.distributions.Categorical(o)
            top1 = p.sample()
            # top1 = [(batch * num_sample)]
            frag = [random.choice(neighbor[j]) if top1[j] == 1 else fragment_dict['i2c'][int(top1[j])]
                    for j in range(top1.shape[0])]
            frag = [convert_smiles_2_mol(f) for f in frag]
            fragment.append(frag)
            input = [convert_mol_2_graph_pyg(f, atom_dict, bond_dict) for f in frag]
            input = Batch.from_data_list(input).to(self.device)
            input = self.fragment_embed(input).unsqueeze(1)
            input = torch.cat([input, cond], dim=-1)
        fragment = list(map(list, zip(*fragment)))
        return fragment


class GraphDecoderFT(nn.Module):
    def __init__(self, num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim, emb_dim, cond_dim_pt, cond_dim,
                 n_rnn_layers, rnn_hid_dim, out_dim, dropout, device, checkpoint):
        super(GraphDecoderFT, self).__init__()
        self.graph_decoder = GraphDecoder(num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim, emb_dim,
                                          cond_dim_pt, n_rnn_layers, rnn_hid_dim, out_dim, dropout, device)
        self.graph_decoder.load_state_dict(checkpoint)
        # for p in self.graph_decoder.parameters():
        #     p.requires_grad = False
        self.ft_layer = nn.Sequential(nn.Linear(cond_dim, 64), nn.LeakyReLU(), nn.Linear(64, cond_dim_pt))

    def forward(self, graph, label, length, cond):
        output = self.graph_decoder(graph, label, length, self.ft_layer(cond))
        return output

    def sample(self, cond, neighbor, max_length, topk, atom_dict, bond_dict, fragment_dict):
        output = self.graph_decoder.sample(self.ft_layer(cond), neighbor, max_length, topk, atom_dict, bond_dict,
                                           fragment_dict)
        return output


class FAME(nn.Module):
    def __init__(self, num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim, latent_dim, emb_dim, out_dim,
                 n_rnn_layers, rnn_hid_dim, dropout, device, latent_dim_pt, encoder_checkpoint, decoder_checkpoint):
        super(FAME, self).__init__()
        self.graph_encoder = JointGraphEncoderFT(num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim,
                                                 latent_dim_pt * 2, latent_dim * 2, dropout, encoder_checkpoint)
        self.graph_decoder = GraphDecoderFT(num_node_features, num_edge_features, n_gnn_layers, gnn_hid_dim, emb_dim,
                                            latent_dim_pt, latent_dim * 2, n_rnn_layers, rnn_hid_dim, out_dim,
                                            dropout, device, decoder_checkpoint)
        self.ge_encoder = GEEncoder(latent_dim)
        self.latent_dim = latent_dim
        self.device = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, graph_drug, graph_frag, batch_frag, ge, label):
        # graph_drug = Batch.from_data_list(graph_drug).to(self.device)
        # batch_frag = torch.LongTensor([len(g) for g in graph_frag])
        # graph_frag = [frag for x in graph_frag for frag in x]
        # graph_frag = Batch.from_data_list(graph_frag).to(self.device)
        mu, log_var = torch.split(self.graph_encoder(graph_drug, ge), self.latent_dim, -1)
        latent = self.reparameterize(mu, log_var)
        cond = self.ge_encoder(ge)
        cond = torch.cat([latent, cond], dim=1)
        output = self. graph_decoder(graph_frag, label, batch_frag, cond)
        return output, mu, log_var, batch_frag

    def loss(self, recon_seq, label_seq, length, mu, logvar, reduce=True):
        recon_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none') \
            (recon_seq.reshape(-1, recon_seq.shape[-1]), label_seq.reshape(-1))
        recon_loss = (recon_loss.reshape(label_seq.shape[0], -1).sum(dim=-1) / length.to(self.device))
        if reduce:
            recon_loss = recon_loss.mean()
        kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return recon_loss, kld_loss

    def sample(self, cond, neighbor, max_length, num_latent, num_sample_per_latent, topk, atom_dict, bond_dict,
               fragment_dict):
        # cond = [batch * num_gene]
        batch, _ = cond.shape
        cond = self.ge_encoder(cond)
        # cond = [batch * latent_dim]
        num_sample = num_latent * num_sample_per_latent
        cond = cond.repeat(1, 1, num_sample).reshape(batch, num_sample, -1)
        # cond = [batch * num_sample * latent_dim]
        latent = torch.randn((batch, num_latent, self.latent_dim), dtype=torch.float32). \
            repeat(1, num_sample_per_latent, 1).to(self.device)
        # latent = [batch * num_sample * latent_dim]
        cond = torch.cat([latent, cond], dim=-1)
        # cond = [batch * num_sample * latent_dim]
        neighbor_new = []
        for nb in neighbor:
            neighbor_new += [nb] * num_sample
        fragment = self.graph_decoder.sample(cond, neighbor_new, max_length, topk, atom_dict, bond_dict, fragment_dict)
        smiles = []
        for frag in fragment:
            try:
                s = merge_fragments(frag)
            except:
                s = 'None'
            smiles.append(s)
        smiles_new = []
        for i in range(batch):
            smiles_new.append(smiles[(i*num_sample):((i+1)*num_sample)])
        return smiles_new
