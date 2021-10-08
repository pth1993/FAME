import numpy as np
import random
from utils import GeneMoleculeDataset, CustomCollateGE, read_pickle, L1000XPRDataset, write_mol_to_file, Metric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CFAMEFT
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from fcd_torch import FCD
from torch_geometric.nn import DataParallel

start_time = datetime.now()

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

atom_file = 'data/atom_dict.pkl'
bond_file = 'data/bond_dict.pkl'
fragment_file = 'data/fragment_dict.pkl'
n_epoch = 50
model_name = 'CFAME'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multi_gpu = False
log_file = 'log_%s.csv' % model_name

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
data_loader_train = DataLoader(data_train, batch_size=128, shuffle=True, collate_fn=CustomCollateGE())
data_val = GeneMoleculeDataset(fragment_file='data/lincs/lincs_fragment_val.csv',
                               ge_file='data/lincs/signature_mol_val_cl.csv',
                               atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
                               label_file='data/lincs/lincs_label_val.pkl')
data_loader_val = DataLoader(data_val, batch_size=128, shuffle=False, collate_fn=CustomCollateGE())
data_test = GeneMoleculeDataset(fragment_file='data/lincs/lincs_fragment_test.csv',
                                ge_file='data/lincs/signature_mol_test_cl.csv',
                                atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
                                label_file='data/lincs/lincs_label_test.pkl',
                                neighbor_file='data/lincs/lincs_mol_test_frag_neighbor.pkl')
data_loader_test = DataLoader(data_test, batch_size=8, shuffle=False, collate_fn=CustomCollateGE())

# data_train = GeneMoleculeDataset(fragment_file='data/lincs/lincs_fragment_train.csv',
#                                  ge_file='data/lincs/signature_mol_train_cl.csv',
#                                  atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
#                                  label_file='data/lincs/lincs_label_train.pkl')
# trained_smiles = set(data_train.smiles_drug)
# data_loader_train = DataLoader(data_train, batch_size=4, shuffle=True, collate_fn=CustomCollateGE())
# data_val = GeneMoleculeDataset(fragment_file='data/lincs/lincs_fragment_val.csv',
#                                ge_file='data/lincs/signature_mol_val_cl.csv',
#                                atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
#                                label_file='data/lincs/lincs_label_val.pkl')
# data_loader_val = DataLoader(data_val, batch_size=4, shuffle=False, collate_fn=CustomCollateGE())
# data_test = GeneMoleculeDataset(fragment_file='data/lincs/lincs_fragment_test.csv',
#                                 ge_file='data/lincs/signature_mol_test_cl.csv',
#                                 atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict,
#                                 label_file='data/lincs/lincs_label_test.pkl',
#                                 neighbor_file='data/lincs/lincs_mol_test_frag_neighbor.pkl')
# data_loader_test = DataLoader(data_test, batch_size=4, shuffle=False, collate_fn=CustomCollateGE())

data_test_ext = L1000XPRDataset(data_file='data/lincs/signature_xpr_cl.csv')
data_loader_test_ext = DataLoader(data_test_ext, batch_size=1, shuffle=False)

fcd = FCD(device=device, n_jobs=8)

data_excapedb = read_pickle('data/lincs/excapedb_calculated.pkl')

checkpoint = torch.load('saved_model/pt8/FAME_0.ckpt', map_location=device)
model = CFAMEFT(num_node_features=len(atom_dict['c2i']), num_edge_features=len(bond_dict['c2i']), n_gnn_layers=5,
                gnn_hid_dim=64, latent_dim=32, emb_dim=32, out_dim=len(fragment_dict['c2i']),
                n_rnn_layers=2, rnn_hid_dim=32, dropout=0.1, device=device, latent_dim_pt=64,
                encoder_checkpoint=checkpoint['encoder_state_dict'], decoder_checkpoint=checkpoint['decoder_state_dict'])

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

if multi_gpu:
    model = nn.DataParallel(model)
model.to(device)

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

# for e in range(n_epoch):
#     model.train()
#     train_loss = 0
#     train_nll_loss = 0
#     train_kld_loss = 0
#     for i, b in enumerate(tqdm(data_loader_train)):
#         graph_drug = b['graph_drug'].to(device)
#         graph_frag = b['graph_frag'].to(device)
#         batch_frag = b['batch_frag']
#         ge = b['ge'].to(device)
#         label = b['label'].to(device)
#         output, mu, logvar, batch_frag = model.forward(graph_drug, graph_frag, batch_frag, ge, label)
#         if isinstance(model, nn.DataParallel):
#             nll, kld = model.module.loss(output, label, batch_frag, mu, logvar)
#         else:
#             nll, kld = model.loss(output, label, batch_frag, mu, logvar)
#         w = 0.1
#         loss = nll + w * kld
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         train_nll_loss += nll.item()
#         train_kld_loss += kld.item()
#     print('Train loss: %.4f - NLL loss: %.4f - KLD loss: %.4f' % (train_loss / (i+1), train_nll_loss / (i+1),
#                                                                   train_kld_loss / (i+1)))
#     train_loss_list.append(train_loss / (i+1))
#     train_nll_loss_list.append(train_nll_loss / (i+1))
#     train_kld_loss_list.append(train_kld_loss / (i+1))
#
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0
#         val_nll_loss = 0
#         val_kld_loss = 0
#         for i, b in enumerate(tqdm(data_loader_val)):
#             graph_drug = b['graph_drug'].to(device)
#             graph_frag = b['graph_frag'].to(device)
#             batch_frag = b['batch_frag']
#             ge = b['ge'].to(device)
#             label = b['label'].to(device)
#             output, mu, logvar, batch_frag = model.forward(graph_drug, graph_frag, batch_frag, ge, label)
#             if isinstance(model, nn.DataParallel):
#                 nll, kld = model.module.loss(output, label, batch_frag, mu, logvar)
#             else:
#                 nll, kld = model.loss(output, label, batch_frag, mu, logvar)
#             w = 0.1
#             loss = nll + w * kld
#             val_loss += loss.item()
#             val_nll_loss += nll.item()
#             val_kld_loss += kld.item()
#         print('Val loss: %.4f - NLL loss: %.4f - KLD loss: %.4f' % (val_loss / (i+1), val_nll_loss / (i+1),
#                                                                     val_kld_loss / (i+1)))
#         val_loss_list.append(val_loss / (i+1))
#         val_nll_loss_list.append(val_nll_loss / (i+1))
#         val_kld_loss_list.append(val_kld_loss / (i+1))
#         if (val_loss / (i+1)) < best_val_loss:
#             best_val_loss = val_loss / (i+1)
#             best_epoch = e
#             torch.save({'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel)
#             else model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
#                        'saved_model/ft1/%s_%d.ckpt' % (model_name, best_epoch))
#
#     with torch.no_grad():
#         test_loss = 0
#         test_nll_loss = 0
#         test_kld_loss = 0
#         for i, b in enumerate(tqdm(data_loader_test)):
#             graph_drug = b['graph_drug'].to(device)
#             graph_frag = b['graph_frag'].to(device)
#             batch_frag = b['batch_frag']
#             ge = b['ge'].to(device)
#             label = b['label'].to(device)
#             output, mu, logvar, batch_frag = model.forward(graph_drug, graph_frag, batch_frag, ge, label)
#             if isinstance(model, nn.DataParallel):
#                 nll, kld = model.module.loss(output, label, batch_frag, mu, logvar)
#             else:
#                 nll, kld = model.loss(output, label, batch_frag, mu, logvar)
#             w = 0.1
#             loss = nll + w * kld
#             test_loss += loss.item()
#             test_nll_loss += nll.item()
#             test_kld_loss += kld.item()
#         print('Test loss: %.4f - NLL loss: %.4f - KLD loss: %.4f' % (test_loss / (i+1), test_nll_loss / (i+1),
#                                                                     test_kld_loss / (i+1)))
#         test_loss_list.append(test_loss / (i+1))
#         test_nll_loss_list.append(test_nll_loss / (i+1))
#         test_kld_loss_list.append(test_kld_loss / (i+1))
#
# print("Loss / NLL / KLD on val set by epoch %d (best val epoch): %.4f - %.4f - %.4f"
#       % (best_epoch + 1, val_loss_list[best_epoch], val_nll_loss_list[best_epoch],
#          val_kld_loss_list[best_epoch]))
#
# df = pd.DataFrame(list(zip(train_loss_list, train_nll_loss_list, train_kld_loss_list, val_loss_list,
#                            val_nll_loss_list, val_kld_loss_list)),
#                   columns=['train_loss', 'train_nll_loss', 'train_kld_loss', 'val_loss', 'val_nll_loss', 'val_kld_loss'],
#                   index=np.arange(len(train_loss_list))+1)
# df.to_csv('output/ft1/%s' % log_file)

best_epoch = 2

# Evaluation
if isinstance(model, nn.DataParallel):
    checkpoint = torch.load('saved_model/ft1/%s_%d.ckpt' % (model_name, best_epoch), map_location=device)
    model.module.load_state_dict(checkpoint['model_state_dict'])
else:
    checkpoint = torch.load('saved_model/ft1/%s_%d.ckpt' % (model_name, best_epoch), map_location=device)
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
        if isinstance(model, nn.DataParallel):
            l, _ = model.module.loss(out, label, batch_frag, mu, logvar, False)
            out = model.module.sample(ge, neighbor=neighbor, num_latent=20, num_sample_per_latent=20, max_length=12,
                                      topk=20, atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict)
        else:
            l, _ = model.loss(out, label, batch_frag, mu, logvar, False)
            out = model.sample(ge, neighbor=neighbor, num_latent=20, num_sample_per_latent=20, max_length=12, topk=20,
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
        if isinstance(model, nn.DataParallel):
            out = model.module.sample(ge, neighbor=neighbor, num_latent=70, num_sample_per_latent=70, max_length=12,
                                      topk=20, atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict)
        else:
            out = model.sample(ge, neighbor=neighbor, num_latent=70, num_sample_per_latent=70, max_length=12, topk=20,
                               atom_dict=atom_dict, bond_dict=bond_dict, fragment_dict=fragment_dict)
        output_ext += out
        refs_ext_fp += [data_excapedb[t]['morgan'] for t in target]
        refs_ext_fcd += [data_excapedb[t]['pref'] for t in target]

write_mol_to_file(output, 'output/ft1/mol_test_%s.csv' % model_name)
write_mol_to_file(output_ext, 'output/ft1/mol_test_ext_%s.csv' % model_name)
output_score = metric.calculate_metric_internal(refs, output, trained_smiles, loss, fcd)
output_score['external_fcd'], output_score['external_jac'] = \
    metric.calculate_metric_external(refs_ext_fcd, refs_ext_fp, output_ext, fcd)
print(output_score)

end_time = datetime.now()
print(end_time - start_time)