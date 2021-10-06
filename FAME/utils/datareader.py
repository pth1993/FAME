from .data_utils import *
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Batch


class MoleculeDataset(Dataset):
    def __init__(self, data_file, atom_dict, bond_dict, fragment_dict, label_file):
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict
        self.fragment_dict = fragment_dict
        self.label, self.smiles_drug, self.smiles_frag, self.mol_drug, self.mol_frag = [], [], [], [], []
        for lb_file in label_file:
            self.label += read_pickle(lb_file)
        for dt_file in data_file:
            smiles_drug, smiles_frag, mol_drug, mol_frag = read_fragment_data(dt_file, return_mol=True)
            self.smiles_drug += smiles_drug
            self.smiles_frag += smiles_frag
            self.mol_drug += mol_drug
            self.mol_frag += mol_frag

    def __len__(self):
        return len(self.smiles_drug)

    def __getitem__(self, idx):
        try:
            self.graph_drug = convert_mol_2_graph_pyg(self.mol_drug[idx], self.atom_dict, self.bond_dict)
            self.graph_frag = [convert_mol_2_graph_pyg(frag, self.atom_dict, self.bond_dict) for frag in self.mol_frag[idx]]
        except:
            print(idx)
            self.graph_drug = convert_mol_2_graph_pyg(self.mol_drug[0], self.atom_dict, self.bond_dict)
            self.graph_frag = [convert_mol_2_graph_pyg(frag, self.atom_dict, self.bond_dict) for frag in self.mol_frag[0]]
        sample = {'smiles_drug': self.smiles_drug[idx], 'smiles_frag': self.smiles_frag[idx],
                  'mol_drug': self.mol_drug[idx], 'mol_frag': self.mol_frag[idx], 'graph_drug': self.graph_drug,
                  'graph_frag': self.graph_frag, 'label': self.label[idx]}
        return sample


class CustomCollate:
    def __call__(self, batch):
        smiles_drug = [x['smiles_drug'] for x in batch]
        smiles_frag = [x['smiles_frag'] for x in batch]
        mol_drug = [x['mol_drug'] for x in batch]
        mol_frag = [x['mol_frag'] for x in batch]
        graph_drug = [x['graph_drug'] for x in batch]
        graph_drug = Batch.from_data_list(graph_drug)
        batch_frag = torch.LongTensor([len(x['graph_frag']) for x in batch])
        graph_frag = [frag for x in batch for frag in x['graph_frag']]
        graph_frag = Batch.from_data_list(graph_frag)
        label = [x['label'] for x in batch]
        label = [torch.tensor(l, dtype=torch.int64) for l in label]
        label_padded = torch.nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=pad_idx)
        return {'smiles_drug': smiles_drug, 'smiles_frag': smiles_frag, 'mol_drug': mol_drug, 'mol_frag': mol_frag,
                'graph_drug': graph_drug, 'graph_frag': graph_frag, 'batch_frag': batch_frag, 'label': label_padded}
