import numpy as np
import random
from utils import MoleculeDataset, CustomCollate, read_pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FAME
from tqdm import tqdm
from datetime import datetime
import pandas as pd

start_time = datetime.now()

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

data_file = 'data/lincs/lincs_fragment_val.csv'
atom_file = 'data/atom_dict.pkl'
bond_file = 'data/bond_dict.pkl'
fragment_file = 'data/fragment_dict.pkl'
n_epoch = 2
model_name = 'FAME'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multi_gpu = True
log_file = 'log_%s.csv' % model_name

atom_dict = read_pickle(atom_file)
bond_dict = read_pickle(bond_file)
fragment_dict = read_pickle(fragment_file)

data_train = MoleculeDataset(data_file=['data/lincs/lincs_fragment_train.csv', 'data/chembl/chembl_fragment_train.csv'],
                             atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
                             label_file=['data/lincs/lincs_label_train.pkl', 'data/chembl/chembl_label_train.pkl'])
data_loader_train = DataLoader(data_train, batch_size=256, shuffle=True, collate_fn=CustomCollate())
data_val = MoleculeDataset(data_file=['data/lincs/lincs_fragment_val.csv', 'data/chembl/chembl_fragment_val.csv'],
                           atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
                           label_file=['data/lincs/lincs_label_val.pkl', 'data/chembl/chembl_label_val.pkl'])
data_loader_val = DataLoader(data_val, batch_size=256, shuffle=False, collate_fn=CustomCollate())
# data_train = MoleculeDataset(data_file=['data/lincs/lincs_fragment_val.csv'],
#                              atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
#                              label_file=['data/lincs/lincs_label_val.pkl'])
# data_loader_train = DataLoader(data_train, batch_size=64, shuffle=True, collate_fn=CustomCollate())
# data_val = MoleculeDataset(data_file=['data/lincs/lincs_fragment_val.csv'],
#                            atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
#                            label_file=['data/lincs/lincs_label_val.pkl'])
# data_loader_val = DataLoader(data_val, batch_size=64, shuffle=False, collate_fn=CustomCollate())

model = FAME(num_node_features=len(atom_dict['c2i']), num_edge_features=len(bond_dict['c2i']), n_gnn_layers=5,
             gnn_hid_dim=64, latent_dim=64, emb_dim=32, out_dim=len(fragment_dict['c2i']),
             n_rnn_layers=2, rnn_hid_dim=32, dropout=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if multi_gpu:
    model = nn.DataParallel(model)
model.to(device)

train_loss_list = []
val_loss_list = []
train_kld_loss_list = []
val_kld_loss_list = []
train_nll_loss_list = []
val_nll_loss_list = []
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
        label = b['label'].to(device)
        output, mu, logvar = model.forward(graph_drug, graph_frag, batch_frag, label)
        if isinstance(model, nn.DataParallel):
            nll, kld = model.module.loss(output, label, batch_frag, mu, logvar)
        else:
            nll, kld = model.loss(output, label, batch_frag, mu, logvar)
        w = 0.1
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
            label = b['label'].to(device)
            output, mu, logvar = model.forward(graph_drug, graph_frag, batch_frag, label)
            if isinstance(model, nn.DataParallel):
                nll, kld = model.module.loss(output, label, batch_frag, mu, logvar)
            else:
                nll, kld = model.loss(output, label, batch_frag, mu, logvar)
            w = 0.1
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
        torch.save({'encoder_state_dict': model.module.graph_encoder.state_dict() if isinstance(model, nn.DataParallel)
        else model.graph_encoder.state_dict(), 'decoder_state_dict': model.module.graph_decoder.state_dict()
        if isinstance(model, nn.DataParallel) else model.graph_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   'saved_model/pt/%s_%d.ckpt' % (model_name, best_epoch))

print("Loss / NLL / KLD on val set by epoch %d (best val epoch): %.4f - %.4f - %.4f"
      % (best_epoch + 1, val_loss_list[best_epoch], val_nll_loss_list[best_epoch],
         val_kld_loss_list[best_epoch]))

df = pd.DataFrame(list(zip(train_loss_list, train_nll_loss_list, train_kld_loss_list, val_loss_list,
                           val_nll_loss_list, val_kld_loss_list)),
                  columns=['train_loss', 'train_nll_loss', 'train_kld_loss', 'val_loss', 'val_nll_loss', 'val_kld_loss'],
                  index=np.arange(len(train_loss_list))+1)
df.to_csv('output/pt/%s' % log_file)

end_time = datetime.now()
print(end_time - start_time)
