import os
import torch
from transformers import BertTokenizer
import json
# from torch_geometric.data import Dataset
from torch.utils.data import Dataset
import os
import random
from data_provider import unimol_dataset
from data_provider.unimol_dataset import D3Collater, D3Collater_Pro
from data_provider.unimol_dataset import D3Dataset, D3Dataset_Pro
from torch_geometric.data import Dataset
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm


class RetrievalDataset_3DText(Dataset):
    def __init__(self, root, text_max_len, text_aug, unimol_dict, max_atoms, tokenizer, args):
        super(RetrievalDataset_3DText, self).__init__(root)
        self.root_ct = root
        self.text_max_len = args.text_max_len
        target_path = os.path.join(root, 'unimol_mol.lmdb')
        self.d3_dataset = D3Dataset(target_path, unimol_dict, max_atoms)
        self.text_name_list = os.listdir(root+'text/')
        #Need sort by CID
        self.text_name_list = sorted(self.text_name_list, key=lambda x: int(x.split('.')[0].split('_')[1]))
        self.tokenizer = tokenizer

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.text_name_list)

    def __getitem__(self, index):
        return self.get_3d(index)
    
    def get_3d(self, index):
        atom_vec, coordinates, edge_type, dist, smiles = self.d3_dataset[index]
        # load and process text
        text_path = os.path.join(self.root_ct, 'text', self.text_name_list[index])
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
        text = ' '.join(lines) + '\n'
        text, mask = self.tokenizer_text(text)
        return (atom_vec, coordinates, edge_type, dist, smiles), text.squeeze(0), mask.squeeze(0)
    
    def collater(self, batch):
        d3_collater = D3Collater(pad_idx=0)
        d2_collater = Collater([], [])

        conf_batch_raw, text3d_batch_raw, text3d_mask= zip(*batch)
        #conf_batch_raw, text3d_batch_raw, text3d_mask = conf_batch_raw.to(device), text3d_batch_raw.to(device), text3d_mask.to(device)
        padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = d3_collater(conf_batch_raw)
        text3d_tokens = d2_collater(text3d_batch_raw)
        text3d_mask = d2_collater(text3d_mask)
        return (padded_atom_vec, padded_dist, padded_edge_type), text3d_tokens, text3d_mask
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask   
class RetrievalDataset_2DText(Dataset):
    def __init__(self, root, text_max_len, text_aug, tokenizer, args):
        super(RetrievalDataset_2DText, self).__init__(root)
        self.root = root
        self.graph_aug = 'noaug'
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root + 'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = tokenizer  #BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.use_smiles = args.use_smiles

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name, smiles_name = self.graph_name_list[index], self.text_name_list[index], self.smiles_name_list[index]
        assert graph_name[len('graph_'):-len('.pt')] == text_name[len('text_'):-len('.txt')] == smiles_name[len('smiles_'):-len('.txt')], print(graph_name, text_name, smiles_name)

        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        '''
        text = ''
        if self.use_smiles:
            text_path = os.path.join(self.root, 'smiles', smiles_name)
            text = 'This molecule is '
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line = line.strip('\n')
                text += f' {line}'
                if count > 1:
                    break
            text += '. '
        
        text_path = os.path.join(self.root, 'text', text_name)
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text += f' {line}'
            if count > 100:
                break
        text += '\n'
        # para-level
        text, mask = self.tokenizer_text(text)
        '''
        # load and process text
        text_path = os.path.join(self.root, 'text', self.text_name_list[index])
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
        text = ' '.join(lines) + '\n'
        text, mask = self.tokenizer_text(text)
        return data_graph, text.squeeze(0), mask.squeeze(0)  # , index

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask
    
    def collater(self, batch):
        d2_collater = Collater([], [])
        graph_batch_raw, text2d_batch_raw, text2d_mask = zip(*batch)
        graph_batch = d2_collater(graph_batch_raw)
        text2d_batch = d2_collater(text2d_batch_raw)
        text2d_mask = d2_collater(text2d_mask)
        return graph_batch, text2d_batch, text2d_mask

class RetrievalDataset_D2D3(Dataset):
    def __init__(self, root, unimol_dict, max_atoms, args):
        super(RetrievalDataset_D2D3, self).__init__(root)
        self.root_d2d3 = root
        self.d3_dataset_d2d3 = D3Dataset(os.path.join(root, 'unimol_mol.lmdb'), unimol_dict, max_atoms)
        ### Need Sort by CID
        self.graph_name_list_d2d3 = sorted(os.listdir(root+'graph/'), key = lambda x:x.split('.')[0].split('_')[-1])
        assert len(self.d3_dataset_d2d3) == len(self.graph_name_list_d2d3)

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list_d2d3)

    def __getitem__(self, index):
        return self.get_d2d3(index)
    
    def get_d2d3(self, index):
        atom_vec, coordinates, edge_type, dist, smiles = self.d3_dataset_d2d3[index]
        graph_name = self.graph_name_list_d2d3[index]
        graph_path = os.path.join(self.root_d2d3, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        return data_graph, (atom_vec, coordinates, edge_type, dist, smiles)
    def collater(self, batch):
        d3_collater = D3Collater(pad_idx=0)
        d2_collater = Collater([], [])
        graph_batch_d2_raw, conf_batch_d3_raw = zip(*batch)
        graph_batch_d2 = d2_collater(graph_batch_d2_raw)
        padded_atom_vec_mol, padded_coordinates_mol, padded_edge_type_mol, padded_dist_mol, smiles_mol = d3_collater(conf_batch_d3_raw)
        return graph_batch_d2, (padded_atom_vec_mol, padded_dist_mol, padded_edge_type_mol)


        

class RetrievalDataset_MolPro(Dataset):
    def __init__(self, root, unimol_dict, unimol_dict_pro, max_atoms, args):
        super(RetrievalDataset_MolPro, self).__init__(root)
        self.d3_dataset_molpro = D3Dataset(os.path.join(root, 'ligand.lmdb'), unimol_dict, max_atoms)
        self.pro_dataset_molpro = D3Dataset_Pro(os.path.join(root, 'pocket.lmdb'), unimol_dict_pro, max_atoms)
        assert len(self.d3_dataset_molpro) == len(self.pro_dataset_molpro)

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.d3_dataset_molpro)

    def __getitem__(self, index):
        return self.get_molpro(index)
    
    def get_molpro(self, index):
        atom_vec_mol, coordinates_mol, edge_type_mol, dist_mol, smiles = self.d3_dataset_molpro[index]
        atom_vec_pro, coordinates_pro, edge_type_pro, dist_pro, residues= self.pro_dataset_molpro[index]
        return (atom_vec_mol, coordinates_mol, edge_type_mol, dist_mol, smiles), (atom_vec_pro, coordinates_pro, edge_type_pro, dist_pro, residues)

    def collater(self, batch):
        d3_collater = D3Collater(pad_idx=0)
        pro_collater = D3Collater_Pro(pad_idx=0)
        mol_batch_raw, pro_batch_raw = zip(*batch)
        padded_atom_vec_mol, padded_coordinates_mol, padded_edge_type_mol, padded_dist_mol, smiles_mol = d3_collater(mol_batch_raw)
        padded_atom_vec_pro, padded_coordinates_pro, padded_edge_type_pro, padded_dist_pro, residues = pro_collater(pro_batch_raw)
        return (padded_atom_vec_mol, padded_dist_mol, padded_edge_type_mol), (padded_atom_vec_pro, padded_dist_pro, padded_edge_type_pro)

