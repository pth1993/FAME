import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem


dummy = Chem.MolFromSmiles('[*]')


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


def convert_smiles_2_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=False)
    return mol


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count


def mol_from_smiles(smi):
    smi = canonicalize(smi)
    return Chem.MolFromSmiles(smi)


def mol_to_smiles(mol):
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)


def strip_dummy_atoms(mol):
    hydrogen = mol_from_smiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    mol = Chem.RemoveHs(mols[0])
    return mol


def strip_dummy_atom(mol):
    hydrogen = mol_from_smiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=False)
    mol = Chem.RemoveHs(mols[0])
    return mol


def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def join_molecules(molA, molB):
    marked, neigh = None, None
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            neigh = atom.GetNeighbors()[0]
            break
    neigh = 0 if neigh is None else neigh.GetIdx()
    if marked is not None:
        ed = Chem.EditableMol(molA)
        ed.RemoveAtom(marked)
        molA = ed.GetMol()
    joined = Chem.ReplaceSubstructs(
        molB, dummy, molA,
        replacementConnectionPoint=neigh,
        useChirality=False)[0]
    Chem.Kekulize(joined)
    return joined


def reconstruct(frags):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags
    try:
        if count_dummies(frags[0]) != 1:
            return None, None
        if count_dummies(frags[-1]) != 1:
            return None, None
        for frag in frags[1:-1]:
            if count_dummies(frag) != 2:
                return None, None
        mol = join_molecules(frags[0], frags[1])
        for i, frag in enumerate(frags[2:]):
            mol = join_molecules(mol, frag)
        mol_to_smiles(mol)
        return mol, frags
    except Exception:
        return None, None


def merge_fragments(frags):
    for i, frag in enumerate(frags):
        if i == 0:
            if count_dummies(frag) == 0:
                return mol_to_smiles(frag)
            elif count_dummies(frag) == 2:
                mol = strip_dummy_atom(frag)
            else:
                mol = frag
        else:
            if count_dummies(frag) == 1:
                mol = join_molecules(mol, frag)
                return mol_to_smiles(mol)
            elif count_dummies(frag) == 2:
                mol = join_molecules(mol, frag)
            else:
                pass
    mol = strip_dummy_atoms(mol)
    return mol_to_smiles(mol)
