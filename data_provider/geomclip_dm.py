# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from data_provider.geomclip_dataset import GeomClipDataset
from torch.utils.data import Dataset, DataLoader
from data_provider.unimol_dataset import D3Collater, D3Collater_Pro
from torch_geometric.loader.dataloader import Collater
from unicore.data import Dictionary
from data_provider.retrival_dataset_multi import  RetrievalDataset_3DText, RetrievalDataset_2DText, RetrievalDataset_MolPro


class MyCollater:
    def __init__(self, tokenizer, text_max_len, pad_idx, load_3d=False):
        self.pad_idx = pad_idx
        self.load_3d = load_3d
        self.d3_collater = D3Collater(pad_idx)
        self.pro_collater = D3Collater_Pro(pad_idx)
        self.d2_collater = Collater([], [])
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def __call__(self, batch):

        pair_ct_list = batch
        conf_batch_raw, text3d_batch_raw, text_3d_mask= zip(*pair_ct_list)

        text3d_tokens = self.d2_collater(text3d_batch_raw)
        text3d_mask = self.d2_collater(text_3d_mask)

        padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(conf_batch_raw)


        return (padded_atom_vec, padded_dist, padded_edge_type), text3d_tokens, text3d_mask


class GeomClipDM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root3d: str = 'data/3M3d_data',
        text_max_len: int = 128,
        dictionary = None,
        dictionary_pro = None,
        tokenizer=None,
        args=None
    ):
        super().__init__()
        self.batch_size = batch_size

        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.dictionary = dictionary
        self.dictionary_pro = dictionary_pro
        self.tokenizer = tokenizer
        self.args = args
    
        self.train_dataset = GeomClipDataset(root3d+'/pretrain/', text_max_len, args.text_aug, dictionary, dictionary_pro, args.unimol_max_atoms)
        self.val_dataset = GeomClipDataset(root3d+'/valid/', text_max_len, args.text_aug, dictionary, dictionary_pro, args.unimol_max_atoms) 
        self.test_dataset = GeomClipDataset(root3d + '/valid/', text_max_len, args.text_aug, dictionary, dictionary_pro, args.unimol_max_atoms)

        self.val_dataset_match_3dtext = RetrievalDataset_3DText(root3d + '/valid/', text_max_len, args.text_aug, dictionary, args.unimol_max_atoms, tokenizer, args).shuffle()  
        self.test_dataset_match_3dtext = RetrievalDataset_3DText(root3d + '/test/', text_max_len, args.text_aug, dictionary, args.unimol_max_atoms, tokenizer, args).shuffle()
        


        self.val_match_loader_3dtext = DataLoader(self.val_dataset_match_3dtext, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_3dtext.collater)
        self.test_match_loader_3dtext = DataLoader(self.test_dataset_match_3dtext, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=self.val_dataset_match_3dtext.collater)

    
    def load_unimol_dict(self):
        dictionary = Dictionary.load('./data/unimol_dict_mol.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        return dictionary
    def load_unimol_pro_dict(self):
        dictionary = Dictionary.load('./data/unimol_dict_pro.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        return dictionary

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.dictionary.pad(), self.args.use_3d)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.dictionary.pad(), self.args.use_3d)
        )

        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=96)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root3d', type=str, default='./data/GeomCLIP_Dataset')

        parser.add_argument('--use_3d', action='store_true', default=True)
        parser.add_argument('--text_max_len', type=int, default=256)
        parser.add_argument('--graph_aug', type=str, default='dnodes')
        parser.add_argument('--text_aug', action='store_true', default=False)
        parser.add_argument('--use_phy_eval', action='store_true', default=False)
        return parent_parser
    