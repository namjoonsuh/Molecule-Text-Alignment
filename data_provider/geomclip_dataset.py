import torch
import json
# from torch_geometric.data import Dataset
from torch.utils.data import Dataset
import os
import random
#from prepare_unimol_dataset import write_lmdb, write_lmdb_multiprocess
from data_provider import unimol_dataset
from data_provider.unimol_dataset import D3Dataset, D3Dataset_Pro
from tqdm import tqdm


class GeomClipDataset(Dataset):
    def __init__(self, root_ct, text_max_len, text_aug, unimol_dict=None, unimol_dict_pro=None, max_atoms=256, prompt='', return_prompt=False):
        super(GeomClipDataset, self).__init__()
        self.prompt = prompt
        self.return_prompt = return_prompt

        self.root_ct = root_ct
        
        self.text_aug = text_aug
        self.text_max_len = text_max_len

        self.text_name_list_3d = os.listdir(root_ct+'text/')
        self.text_name_list_3d=sorted(self.text_name_list_3d, key=lambda x: int(x.split('.')[0].split('_')[1]))
        self.tokenizer = None
        target_path = os.path.join(root_ct, 'unimol_mol.lmdb')
        self.d3_dataset = D3Dataset(target_path, unimol_dict, max_atoms)
        assert len(self.d3_dataset) == len(self.text_name_list_3d),print(len(self.d3_dataset),len(self.text_name_list_3d))


        self.permutation = None
    
    def shuffle(self):
        self.permutation = torch.randperm(len(self)).numpy()
        return self

    def __len__(self):
        return len(self.text_name_list_3d)

    def get_2d(self, index):
        if index >= len(self.text_name_list_2d):
            index = index % len(self.text_name_list_2d)
        graph_name, text_name = self.graph_name_list_2d[index], self.text_name_list_2d[index]
        # load and process graph
        graph_path = os.path.join(self.root_gt, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root_gt, 'text', text_name)
        if self.text_aug:
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                text_list.append(line.strip('\n') + '\n')
                if count > 100:
                    break
            text_sample = random.sample(text_list, 1)
            text_list.clear()
            text, mask = self.tokenizer_text(text_sample[0])
        else:
            with open(text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip()]
            text = ' '.join(lines) + '\n'
            text, mask = self.tokenizer_text(text)
        
        assert not self.return_prompt
        return data_graph, text.squeeze(0), mask.squeeze(0)
    

    def get_3d(self, index):
        # To hadle data size of diffretnt alignment not equal
        if index >= len(self.text_name_list_3d):
            index = index % len(self.text_name_list_3d)
        atom_vec, coordinates, edge_type, dist, smiles = self.d3_dataset[index]
        # load and process text
        text_path = os.path.join(self.root_ct, 'text', self.text_name_list_3d[index])
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
        text = ' '.join(lines) + '\n'
        text, mask = self.tokenizer_text(text)

        ## load smiles
        #smiles_path = os.path.join(self.root_ct, 'smiles', self.smiles_name_list_3d[index])
        #with open(smiles_path, 'r') as f:
        #    loaded_smiles = f.readline().strip()
        ## check if the loaded smiles match the one returned by d3_dataset
        
        #assert loaded_smiles == smiles, print(loaded_smiles, smiles)
        # if self.return_prompt:
        #     smiles_prompt = self.prompt.format(smiles[:72])
        #     # if self.root.find('test') >= 0:
        #     #     smiles_prompt += "The molecule is "
        #     return (atom_vec, coordinates, edge_type, dist, smiles), smiles_prompt, text, index
        return (atom_vec, coordinates, edge_type, dist, smiles), text.squeeze(0), mask.squeeze(0)

    def __getitem__(self, index):
        ## consider the permutation
        if self.permutation is not None:
            index = self.permutation[index]
        pair_ct = self.get_3d(index)
        return pair_ct

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask
