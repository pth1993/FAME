import numpy as np
import pandas as pd
import json
import ast
import pickle
from rdkit import Chem
from torch_geometric.data import Data
import torch
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

pad_idx = 0
max_length = 12
target_list = ['SMAD3', 'TP53', 'EGFR', 'AKT1', 'AURKB', 'CTSK', 'MTOR', 'AKT2', 'PIK3CA', 'HDAC1']


class MolData(Data):
    def __init__(self, edge_index=None, x=None, edge_attr=None):
        super().__init__()
        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == ('edge_index' or 'edge_attr'):
            return 1
        else:
            return 0


def read_pickle(input_file):
    with open(input_file, 'rb') as f:
        output = pickle.load(f)
    return output


def read_lincs_mol_data(input_file):
    data = pd.read_csv(input_file)
    data = data[data.pert_iname != filter]
    gene_expression = data['gene_expression'].tolist()
    gene_expression = [ast.literal_eval(g) for g in gene_expression]
    return gene_expression


def read_lincs_xpr_data(input_file):
    data = pd.read_csv(input_file)
    target = data['pert_iname'].tolist()
    gene_expression = data['gene_expression'].tolist()
    gene_expression = [ast.literal_eval(g) for g in gene_expression]
    return target, gene_expression


def read_excapedb_data(excapedb_file, mol_file, fcd):
    output = dict()
    data = pd.read_csv(excapedb_file)
    with open(mol_file) as f:
        data_mol = json.load(f)
    for t in target_list:
        output[t] = dict()
        output[t]['smiles'] = data[data['target'].isin([t])]['smiles'].tolist()
        output[t]['morgan'] = np.array([data_mol[s]['morgan'] for s in output[t]['smiles']])
        output[t]['pref'] = fcd.precalc(output[t]['smiles'])
    return output


def write_mol_to_file(smiles, output_file):
    with open(output_file, 'w') as f:
        if isinstance(smiles[0], list):
            smiles = [s for sm in smiles for s in sm]
        f.write('\n'.join(smiles))


def convert_smiles_2_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol


def convert_mol_2_graph_pyg(mol, atom_dict, bond_dict):
    if mol is not None:
        bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()] for b in mol.GetBonds()]
        bonds_begin, bonds_end, bonds_type = zip(*bonds)
        edge_index = torch.tensor([bonds_begin, bonds_end], dtype=torch.long)
        edge_attr = np.zeros((len(bonds), len(bond_dict['i2c'])))
        for i, b_type in enumerate(bonds_type):
            edge_attr[i, bond_dict['c2i'][b_type]] = 1
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        x = np.zeros((mol.GetNumAtoms(), len(atom_dict['i2c'])), dtype=np.float)
        atoms = [atom_dict['c2i'][atom.GetAtomicNum()] for atom in mol.GetAtoms()]
        for i, a_type in enumerate(atoms):
            x[i, a_type] = 1
        x = torch.tensor(x, dtype=torch.float32)
        graph = MolData(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return graph


def read_fragment_data(input_file, return_mol=False, max_length=12):
    data = pd.read_csv(input_file).dropna()
    smiles_drug = data['smiles'].tolist()
    smiles_frag = [ast.literal_eval(i) for i in data['fragments'].tolist()]
    smiles_drug_new = []
    smiles_frag_new = []
    for s_d, s_f in zip(smiles_drug, smiles_frag):
        if len(s_f) <= max_length:
            smiles_drug_new.append(s_d)
            smiles_frag_new.append(s_f)
    smiles_drug = smiles_drug_new
    smiles_frag = smiles_frag_new
    if return_mol:
        mol_drug = [convert_smiles_2_mol(s) for s in smiles_drug]
        mol_frag = [[convert_smiles_2_mol(s) for s in smi] for smi in smiles_frag]
    else:
        mol_drug = None
        mol_frag = None
    return smiles_drug, smiles_frag, mol_drug, mol_frag


def read_mol_from_file(input_file, num_sample):
    data = []
    with open(input_file) as f:
        for line in f:
            data.append(line.strip())
    data = [data[i:(i+num_sample)] for i in range(0, len(data), num_sample)]
    return data
