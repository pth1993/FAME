from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Fraggle import FraggleSim
from rdkit import RDLogger
import numpy as np
import torch
from tqdm import tqdm
import math
from scipy.spatial import distance



lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class Metric(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def smiles_sanitize(self, smiles):
        mol = [Chem.MolFromSmiles(s) for s in smiles]
        smiles = [Chem.MolToSmiles(m) if m is not None else None for m in mol]
        return {'smiles': smiles, 'mol': mol, 'smiles_set': set(smiles)}

    def calculate_valid(self, smiles):
        return np.mean([1 if s is not None else 0 for s in smiles])

    def calculate_unique(self, smiles, smiles_set):
        return len(smiles_set - {None}) / np.sum([1 if s is not None else 0 for s in smiles])

    def calculate_novel(self, reference_smiles_set, generated_smiles_set):
        generated_smiles_set = generated_smiles_set - {None}
        reference_smiles_set = reference_smiles_set - {None}
        return 1 - len(reference_smiles_set.intersection(generated_smiles_set)) / len(generated_smiles_set)

    def calculate_metric_internal(self, reference_smiles, generated_smiles, trained_smiles, nll_loss, fcd):
        # reference_smiles = [num_data]
        # generated_smiles = [num_data * num_sample]
        generated_smiles = [o for out in generated_smiles for o in out]
        # generated_smiles = [(num_data * num_sample)]
        num_sample = len(generated_smiles) / len(reference_smiles)
        print('Sanitize smiles...')
        reference_data = self.smiles_sanitize(reference_smiles)
        generated_data = self.smiles_sanitize(generated_smiles)
        if generated_data['smiles_set'] == {None}:
            return {'valid': 0 if self.config['valid'] else 'N/A',
                    'novel': 0 if self.config['novel'] else 'N/A',
                    'unique': 0 if self.config['unique'] else 'N/A'}
        print('Generated smiles set is not None.')
        print('Calculate Valid...')
        valid = self.calculate_valid(generated_data['smiles']) if self.config['valid'] else 'N/A'
        print('Calculate Novel...')
        novel = self.calculate_novel(trained_smiles, generated_data['smiles_set']) \
            if self.config['novel'] else 'N/A'
        print('Calculate Unique...')
        unique = self.calculate_unique(generated_data['smiles'], generated_data['smiles_set']) \
            if self.config['unique'] else 'N/A'
        print('Get Fingerprint...')
        self.get_fingerprint_reference(reference_data)
        self.get_fingerprint_sample(generated_data, num_sample)
        print('Calculate Internal Diversity...')
        din = self.calculate_diversity_in(generated_data['fp_morgan']) if self.config['din'] else 'N/A'
        if isinstance(nll_loss, list):
            nll_loss = np.mean(nll_loss) if self.config['nll'] else 'N/A'
        else:
            nll_loss = nll_loss if self.config['nll'] else 'N/A'
        print('Calculate Internal FCD...')
        smiles_sanitized = [s for s in generated_data['smiles'] if s]
        smiles_sanitized = [s for s in self.smiles_sanitize(smiles_sanitized)['smiles'] if s]
        fcd_socre = fcd(gen=smiles_sanitized, ref=reference_data['smiles']) if self.config['int_fcd'] else 'N/A'
        return {'valid': valid, 'novel': novel, 'unique': unique, 'internal diversity': din, 'NLL': nll_loss,
                'internal fcd': fcd_socre}

    def calculate_metric_external(self, reference_fcd, reference_fp, generated_smiles, fcd):
        # generated_smiles = [num_data * num_sample]
        print('Calculate External FCD...')
        fcd_score = []
        jac_score = []
        for ref_fcd, ref_fp, gen_smiles in tqdm(zip(reference_fcd, reference_fp, generated_smiles)):
            smiles_sanitized = [s for s in self.smiles_sanitize(gen_smiles)['smiles'] if s]
            if len(smiles_sanitized) > 1:
                smiles_sanitized = [s for s in self.smiles_sanitize(smiles_sanitized)['smiles'] if s]
                fcd_s = fcd(gen=smiles_sanitized, pref=ref_fcd)
                # if math.isnan(sc):
                #     print(smiles_sanitized)
                fcd_score.append(fcd_s)
                gen_data = {'smiles': smiles_sanitized}
                self.get_fingerprint_reference(gen_data)
                jac_score.append(self.calculate_nearest_distance(ref_fp, gen_data['fp_morgan']))
            # else:
            #     print('No valid smiles')
        return np.mean(fcd_score), np.mean(jac_score)

    def get_fingerprint_reference(self, data):
        fps_morgan = []
        fps_maccs = []
        for s in data['smiles']:
            if s is not None:
                m = Chem.MolFromSmiles(s)
                fps_morgan.append(list(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)))
                fps_maccs.append(list(MACCSkeys.GenMACCSKeys(m)))
            else:
                pass
        data['fp_morgan'] = fps_morgan
        data['fp_maccs'] = fps_maccs

    def get_fingerprint_sample(self, data, num_sample):
        fps_morgan_dict = dict()
        fps_morgan_per_example = []
        fps_morgan = []
        fps_maccs_dict = dict()
        fps_maccs_per_example = []
        fps_maccs = []
        smiles_per_sample = self.reshape_per_example(data['smiles'], len(data['smiles']) / num_sample)
        for s in (data['smiles_set'] - {None}):
            m = Chem.MolFromSmiles(s)
            try:
                morgan = list(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
                maccs = list(MACCSkeys.GenMACCSKeys(m))
                fps_morgan_dict[s] = morgan
                fps_maccs_dict[s] = maccs
            except:
                print('Cannot get fingerprint: %s' % s)
        for input_str in smiles_per_sample:

            input_str_set = set(input_str)
            fp_morgan = []
            fp_maccs = []
            for s in input_str_set:
                if s in fps_morgan_dict:
                    fp_morgan.append(fps_morgan_dict[s])
                    fp_maccs.append(fps_maccs_dict[s])
            fps_morgan_per_example.append(fp_morgan)
            fps_maccs_per_example.append(fp_maccs)

            fp_morgan = []
            fp_maccs = []
            for s in input_str:
                if s in fps_morgan_dict:
                    fp_morgan.append(fps_morgan_dict[s])
                    fp_maccs.append(fps_maccs_dict[s])
            fps_morgan.append(fp_morgan)
            fps_maccs.append(fp_maccs)
        data['fp_morgan'] = fps_morgan
        data['fp_maccs'] = fps_maccs
        data['fp_morgan_per_sample'] = fps_morgan_per_example
        data['fp_maccs_per_sample'] = fps_maccs_per_example

    def reshape_per_example(self, input_string, factor):
        input_string = np.array(input_string)
        output_string = np.split(input_string, factor)
        return output_string

    def calculate_diversity_in(self, input_fps):
        input_fps = np.array([i for sl in input_fps for i in sl])
        return 1 - (self.average_agg_tanimoto(input_fps, input_fps, 5000, agg='mean')).mean()

    def calculate_nearest_distance(self, ref_fp, gen_fp):
        d = distance.cdist(gen_fp, ref_fp, 'jaccard')
        return np.min(d, axis=1).mean()

    def average_agg_tanimoto(self, ref_fp, gen_fp, batch_size, agg, p=1):
        assert agg in ['max', 'mean'], "Can aggregate only max or mean"
        agg_tanimoto = np.zeros(len(gen_fp))
        total = np.zeros(len(gen_fp))
        for j in range(0, ref_fp.shape[0], batch_size):
            x_stock = torch.tensor(ref_fp[j:j + batch_size]).to(self.device).float()
            for i in range(0, gen_fp.shape[0], batch_size):
                y_gen = torch.tensor(gen_fp[i:i + batch_size]).to(self.device).float()
                y_gen = y_gen.transpose(0, 1)
                tp = torch.mm(x_stock, y_gen)
                jac = (tp / (x_stock.sum(1, keepdim=True) +
                             y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
                jac[np.isnan(jac)] = 1
                if p != 1:
                    jac = jac**p
                if agg == 'max':
                    agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                        agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
                elif agg == 'mean':
                    agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                    total[i:i + y_gen.shape[1]] += jac.shape[0]
        if agg == 'mean':
            agg_tanimoto /= total
        if p != 1:
            agg_tanimoto = (agg_tanimoto)**(1/p)
        return np.mean(agg_tanimoto)

    #
    # def calculate_diversity_ex(self, reference_fps, sample_fps):
    #     return np.mean(np.min(cdist(reference_fps, sample_fps, 'jaccard'), axis=1))
    #
    # def calculate_similarity_loss(self, reference_fps, sample_fps_per_example):
    #     score = []
    #     for i, r in enumerate(reference_fps):
    #         if len(sample_fps_per_example[i]) == 0:
    #             score.append(0.0)
    #         else:
    #             score.append(np.max(1 - cdist([r], sample_fps_per_example[i], 'jaccard')))
    #     return np.mean(score)
    #
    # def calculate_similarity_fraggle(self, reference_mol, sample_mol, num_sample):
    #     sample_mol_per_example = self.reshape_per_example(sample_mol, len(sample_mol) / num_sample)
    #     score = []
    #     for i, r in enumerate(reference_mol):
    #         sc = []
    #         try:
    #             FraggleSim.GetFraggleSimilarity(r, r)
    #             for s in sample_mol_per_example[i]:
    #                 if s is not None:
    #                     try:
    #                         sc.append(FraggleSim.GetFraggleSimilarity(r, s)[0])
    #                     except:
    #                         # print('Cannot get fraggle for sample')
    #                         pass
    #             if len(sc) == 0:
    #                 score.append(0.0)
    #             else:
    #                 score.append(np.max(sc))
    #         except:
    #             # print('Cannot get fraggle for reference')
    #             pass
    #     return np.mean(score)
    #
    # def calculate_metric(self, reference_string, sample_string):
    #     num_sample = len(sample_string) / len(reference_string)
    #     print('Smiles to Mol')
    #     reference_mol = self.smiles2mol(reference_string)
    #     sample_mol = self.smiles2mol(sample_string)
    #     print('Smiles to Canonical Smiles')
    #     reference_string = self.get_canon_smiles(reference_mol)
    #     sample_string = self.get_canon_smiles(sample_mol)
    #     print('Smiles to Fingerprint')
    #     reference_fps_morgan, reference_fps_maccs = self.get_fingerprint_reference(reference_string)
    #     sample_fps_morgan, sample_fps_morgan_per_example, sample_fps_maccs, sample_fps_maccs_per_example = \
    #         self.get_fingerprint_sample(sample_string, num_sample)
    #     if set(sample_string) == {None}:
    #         return {'valid': 0, 'novel': 0, 'unique': 0, 'din': 0, 'dex': 0, 'supervised morgan': 0,
    #                 'supervised maccs': 0, 'supervised fraggle': 0}
    #     print('Calculate Valid')
    #     if self.config['valid']:
    #         valid = self.calculate_valid(sample_string)
    #     else:
    #         valid = 'N/A'
    #     print('Calculate Novel')
    #     if self.config['novel']:
    #         novel = self.calculate_novel(reference_string, sample_string)
    #     else:
    #         novel = 'N/A'
    #     print('Calculate Unique')
    #     if self.config['unique']:
    #         unique = self.calculate_unique(sample_string)
    #     else:
    #         unique = 'N/A'
    #     print('Calculate din')
    #     if self.config['din']:
    #         din = self.calculate_diversity_in(sample_fps_morgan)
    #     else:
    #         din = 'N/A'
    #     print('Calculate dex')
    #     if self.config['dex']:
    #         dex = self.calculate_diversity_ex(reference_fps_morgan, sample_fps_morgan)
    #     else:
    #         dex = 'N/A'
    #     print('Calculate morgan')
    #     if self.config['morgan']:
    #         sim_morgan = self.calculate_similarity_loss(reference_fps_morgan, sample_fps_morgan_per_example)
    #     else:
    #         sim_morgan = 'N/A'
    #     print('Calculate maccs')
    #     if self.config['maccs']:
    #         sim_maccs = self.calculate_similarity_loss(reference_fps_maccs, sample_fps_maccs_per_example)
    #     else:
    #         sim_maccs = 'N/A'
    #     print('Calculate fraggle')
    #     if self.config['fraggle']:
    #         sim_fraggle = self.calculate_similarity_fraggle(reference_mol, sample_mol, num_sample)
    #     else:
    #         sim_fraggle = 'N/A'
    #     return {'valid': valid, 'novel': novel, 'unique': unique, 'din': din, 'dex': dex,
    #             'supervised morgan': sim_morgan, 'supervised maccs': sim_maccs, 'supervised fraggle': sim_fraggle}
