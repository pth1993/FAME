import numpy as np
import random
from utils import GeneMoleculeDataset, CustomCollateGE, read_pickle, L1000XPRDataset, write_mol_to_file, Metric
import torch
from torch.utils.data import DataLoader
from model import FAME
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from fcd_torch import FCD

start_time = datetime.now()

# random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# io parameters
atom_file = 'data/atom_dict.pkl'
bond_file = 'data/bond_dict.pkl'
fragment_file = 'data/fragment_dict.pkl'
model_name = 'FAME'
log_file = 'log_%s.csv' % model_name

# training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_epoch = 50
batch_size_train = 128
batch_size_test_int = 8
batch_size_test_ext = 1
lr_rate = 0.0001

# model parameters
n_gnn_layers = 5
gnn_hid_dim = 64
latent_dim = 32
emb_dim = 32
n_rnn_layers = 2
rnn_hid_dim = 32
dropout = 0.1
latent_dim_pt = 64
w = 0.1

# sampling parameters
num_latent_int = 20
num_sample_per_latent_int = 20
num_latent_ext = 70
num_sample_per_latent_ext = 70
max_length = 12
topk = 20

# read data
atom_dict = read_pickle(atom_file)
bond_dict = read_pickle(bond_file)
fragment_dict = read_pickle(fragment_file)
metric = Metric(config={'valid': True, 'novel': True, 'unique': True, 'din': True, 'nll': True, 'int_fcd': True},
                device=device)
data_train = GeneMoleculeDataset(fragment_file='data/lincs/lincs_fragment_train.csv',
                                 ge_file='data/lincs/signature_mol_train_cl.csv',
                                 atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
                                 label_file='data/lincs/lincs_label_train.pkl')
trained_smiles = set(data_train.smiles_drug)
data_loader_train = DataLoader(data_train, batch_size=batch_size_train, shuffle=True, collate_fn=CustomCollateGE())
data_val = GeneMoleculeDataset(fragment_file='data/lincs/lincs_fragment_val.csv',
                               ge_file='data/lincs/signature_mol_val_cl.csv',
                               atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
                               label_file='data/lincs/lincs_label_val.pkl')
data_loader_val = DataLoader(data_val, batch_size=batch_size_train, shuffle=False, collate_fn=CustomCollateGE())
data_test = GeneMoleculeDataset(fragment_file='data/lincs/lincs_fragment_test.csv',
                                ge_file='data/lincs/signature_mol_test_cl.csv',
                                atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
                                label_file='data/lincs/lincs_label_test.pkl',
                                neighbor_file='data/lincs/lincs_mol_frag_neighbor.pkl')
data_loader_test = DataLoader(data_test, batch_size=batch_size_test_int, shuffle=False, collate_fn=CustomCollateGE())
data_test_ext = L1000XPRDataset(data_file='data/lincs/signature_xpr_cl.csv')
data_loader_test_ext = DataLoader(data_test_ext, batch_size=batch_size_test_ext, shuffle=False)
fcd = FCD(device=device, n_jobs=8)
data_excapedb = read_pickle('data/lincs/excapedb.pkl')

# load model
checkpoint = torch.load('saved_model/pre_trained_weights.ckpt', map_location=device)
model = FAME(num_node_features=len(atom_dict['c2i']), num_edge_features=len(bond_dict['c2i']), n_gnn_layers=n_gnn_layers,
             gnn_hid_dim=gnn_hid_dim, latent_dim=latent_dim, emb_dim=emb_dim, out_dim=len(fragment_dict['c2i']),
             n_rnn_layers=n_rnn_layers, rnn_hid_dim=rnn_hid_dim, dropout=dropout, device=device,
             latent_dim_pt=latent_dim_pt, encoder_checkpoint=checkpoint['encoder_state_dict'],
             decoder_checkpoint=checkpoint['decoder_state_dict'])
model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_rate)

# training
train_loss_list = []
val_loss_list = []
test_loss_list = []
train_kld_loss_list = []
val_kld_loss_list = []
test_kld_loss_list = []
train_nll_loss_list = []
val_nll_loss_list = []
test_nll_loss_list = []
best_val_loss = float('inf')

for e in range(n_epoch):
    model.train()
    train_loss = 0
    train_nll_loss = 0
    train_kld_loss = 0
    for i, b in enumerate(tqdm(data_loader_train)):
        graph_drug = b['graph_drug'].to(device)
        graph_frag = b['graph_frag'].to(device)
        batch_frag = b['batch_frag']
        ge = b['ge'].to(device)
        label = b['label'].to(device)
        output, mu, logvar, batch_frag = model.forward(graph_drug, graph_frag, batch_frag, ge, label)
        nll, kld = model.loss(output, label, batch_frag, mu, logvar)
        loss = nll + w * kld
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_nll_loss += nll.item()
        train_kld_loss += kld.item()
    print('Train loss: %.4f - NLL loss: %.4f - KLD loss: %.4f' % (train_loss / (i+1), train_nll_loss / (i+1),
                                                                  train_kld_loss / (i+1)))
    train_loss_list.append(train_loss / (i+1))
    train_nll_loss_list.append(train_nll_loss / (i+1))
    train_kld_loss_list.append(train_kld_loss / (i+1))

    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_nll_loss = 0
        val_kld_loss = 0
        for i, b in enumerate(tqdm(data_loader_val)):
            graph_drug = b['graph_drug'].to(device)
            graph_frag = b['graph_frag'].to(device)
            batch_frag = b['batch_frag']
            ge = b['ge'].to(device)
            label = b['label'].to(device)
            output, mu, logvar, batch_frag = model.forward(graph_drug, graph_frag, batch_frag, ge, label)
            nll, kld = model.loss(output, label, batch_frag, mu, logvar)
            loss = nll + w * kld
            val_loss += loss.item()
            val_nll_loss += nll.item()
            val_kld_loss += kld.item()
        print('Val loss: %.4f - NLL loss: %.4f - KLD loss: %.4f' % (val_loss / (i+1), val_nll_loss / (i+1),
                                                                    val_kld_loss / (i+1)))
        val_loss_list.append(val_loss / (i+1))
        val_nll_loss_list.append(val_nll_loss / (i+1))
        val_kld_loss_list.append(val_kld_loss / (i+1))
        if (val_loss / (i+1)) < best_val_loss:
            best_val_loss = val_loss / (i+1)
            best_epoch = e
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       'saved_model/%s.ckpt' % model_name)

    with torch.no_grad():
        test_loss = 0
        test_nll_loss = 0
        test_kld_loss = 0
        for i, b in enumerate(tqdm(data_loader_test)):
            graph_drug = b['graph_drug'].to(device)
            graph_frag = b['graph_frag'].to(device)
            batch_frag = b['batch_frag']
            ge = b['ge'].to(device)
            label = b['label'].to(device)
            output, mu, logvar, batch_frag = model.forward(graph_drug, graph_frag, batch_frag, ge, label)
            nll, kld = model.loss(output, label, batch_frag, mu, logvar)
            loss = nll + w * kld
            test_loss += loss.item()
            test_nll_loss += nll.item()
            test_kld_loss += kld.item()
        print('Test loss: %.4f - NLL loss: %.4f - KLD loss: %.4f' % (test_loss / (i+1), test_nll_loss / (i+1),
                                                                    test_kld_loss / (i+1)))
        test_loss_list.append(test_loss / (i+1))
        test_nll_loss_list.append(test_nll_loss / (i+1))
        test_kld_loss_list.append(test_kld_loss / (i+1))

print("Loss / NLL / KLD on val set by epoch %d (best val epoch): %.4f - %.4f - %.4f"
      % (best_epoch + 1, val_loss_list[best_epoch], val_nll_loss_list[best_epoch],
         val_kld_loss_list[best_epoch]))

df = pd.DataFrame(list(zip(train_loss_list, train_nll_loss_list, train_kld_loss_list, val_loss_list,
                           val_nll_loss_list, val_kld_loss_list)),
                  columns=['train_loss', 'train_nll_loss', 'train_kld_loss', 'val_loss', 'val_nll_loss', 'val_kld_loss'],
                  index=np.arange(len(train_loss_list))+1)
df.to_csv('output/%s' % log_file)

# evaluation
checkpoint = torch.load('saved_model/%s.ckpt' % model_name, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    output = []
    refs = []
    loss = []
    for i, b in enumerate(tqdm(data_loader_test)):
        graph_drug = b['graph_drug'].to(device)
        graph_frag = b['graph_frag'].to(device)
        batch_frag = b['batch_frag']
        ge = b['ge'].to(device)
        neighbor = b['neighbor']
        label = b['label'].to(device)
        ref = b['smiles_drug']
        out, mu, logvar, batch_frag = model.forward(graph_drug, graph_frag, batch_frag, ge, label)
        l, _ = model.loss(out, label, batch_frag, mu, logvar, False)
        out = model.sample(ge, neighbor=neighbor, num_latent=num_latent_int,
                           num_sample_per_latent=num_sample_per_latent_int, max_length=max_length, topk=topk,
                           atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict)
        l = l.detach().cpu().tolist()
        loss += l
        output += out
        refs += ref

    output_ext = []
    refs_ext_fp = []
    refs_ext_fcd = []
    for i, b in enumerate(tqdm(data_loader_test_ext)):
        ge = b['ge'].to(device)
        target = b['target']
        out = model.sample(ge, neighbor=neighbor, num_latent=num_latent_ext,
                           num_sample_per_latent=num_sample_per_latent_ext, max_length=max_length, topk=topk,
                           atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict)
        output_ext += out
        refs_ext_fp += [data_excapedb[t]['morgan'] for t in target]
        refs_ext_fcd += [data_excapedb[t]['pref'] for t in target]

write_mol_to_file(output, 'output/int_mol_set_%s.csv' % model_name)
write_mol_to_file(output_ext, 'output/ext_mol_set_%s.csv' % model_name)

output_score = metric.calculate_metric_internal(refs, output, trained_smiles, loss, fcd)
output_score['external_fcd'], output_score['external_jac'] = \
    metric.calculate_metric_external(refs_ext_fcd, refs_ext_fp, output_ext, fcd)
print(output_score)

end_time = datetime.now()
print(end_time - start_time)
