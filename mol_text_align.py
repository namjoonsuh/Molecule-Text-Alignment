from transformers import BertTokenizer, BertConfig, BertLMHeadModel
from torch.nn import functional as F
from unicore.data import Dictionary
from model.unimol_simple import SimpleUniMolModel
from LDMol.utils import AE_SMILES_encoder, regexTokenizer
from LDMol.train_autoencoder import ldmol_autoencoder

import torch, argparse
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def init_tokenizer(bert_name):
    #bert_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer

def init_sci_bert(bert_name):
    #bert_name = 'allenai/scibert_scivocab_uncased'
    encoder_config = BertConfig.from_pretrained(bert_name)
    model = BertLMHeadModel.from_pretrained(bert_name, config=encoder_config).to(device)

    return model

def init_unimol_encoder(args):
    dictionary = Dictionary.load('./unimol_dict_mol.txt')
    dictionary.add_symbol("[MASK]", is_special=True)
    unimol_model = SimpleUniMolModel(args, dictionary).to(device)
    ckpt = torch.load('./mol_pre_no_h_220816.pt', map_location=torch.device(device))['model']
    missing_keys, unexpected_keys = unimol_model.load_state_dict(ckpt, strict=False)

    ln_graph = LayerNorm(unimol_model.num_features)
    return unimol_model, ln_graph, dictionary

def init_smiles_encoder(args):
    # Load pretrained encoder 
    ae_config = {
            'bert_config_decoder': './LDMol/config_decoder.json',
            'bert_config_encoder': './LDMol/config_encoder.json',
            'embed_dim': 256,
        }

    tokenizer = regexTokenizer(vocab_path='./LDMol/vocab_bpe_300_sc.txt', max_len=127) # newtkn
    ae_model = ldmol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer, use_linear=True).to(device)

    if args.vae:
        checkpoint = torch.load(args.vae)
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']
        msg = ae_model.load_state_dict(state_dict, strict=False)
    for param in ae_model.parameters():
        param.requires_grad = False
    return ae_model

def computeNCELoss(embed_A, embed_B, temperature_gt):
    batch_size = embed_A.size(0)
    cos_sim = torch.mm(embed_A, embed_B.transpose(0, 1))
    logits_per_A = cos_sim/temperature_gt
    logits_per_B = logits_per_A.transpose(0,1)

    labels = torch.arange(batch_size, dtype=torch.long).to(device)
    loss_A = F.cross_entropy(logits_per_A, labels)
    loss_B = F.cross_entropy(logits_per_B, labels)
    loss = (loss_A + loss_B)/2
    return loss

class MolTextAligner(nn.Module):

    def __init__(self, args):
        super(MolTextAligner,self).__init__()

        self.temperature = args.temperature
        self.sci_bert = init_sci_bert(args.bert_name)
        self.tokenizer = init_tokenizer(args.bert_name)
        self.sci_bert.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.sci_bert.state_dict()
        for name, param in self.sci_bert.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        self.uni_mol_encoder, _, _ = init_unimol_encoder(args)
        self.ae_model = init_smiles_encoder(args)
        
        self.Smiles_proj = nn.Linear(args.smiles_encoder_dim, args.embed_dim).to(device)
        self.ThreeD_proj = nn.Linear(args.unimol_encoder_embed_dim, args.embed_dim).to(device)
        self.Text_proj = nn.Linear(args.text_encoder_embed_dim, args.embed_dim).to(device)

    def computeNCELoss(embed_A, embed_B, temperature_gt):
        batch_size = embed_A.size(0)
        cos_sim = torch.mm(embed_A, embed_B.transpose(0, 1))
        logits_per_A = cos_sim/temperature_gt
        logits_per_B = logits_per_A.transpose(0,1)

        labels = torch.arange(batch_size, dtype=torch.long)
        loss_A = F.cross_entropy(logits_per_A, labels)
        loss_B = F.cross_entropy(logits_per_B, labels)
        loss = (loss_A + loss_B)/2
        return loss
    
    def forward(self, smiles_tokens, padded_atom_vec, padded_dist, padded_edge_type, token_idxs, attention_masks):
        
        with torch.no_grad():
            batch_Smiles = AE_SMILES_encoder(smiles_tokens, self.ae_model)
            batch_ThreeD, _ = self.uni_mol_encoder(padded_atom_vec, padded_dist, padded_edge_type)
            batch_Text = self.sci_bert.bert(token_idxs, attention_masks, return_dict=True)

        Smiles_embed = self.Smiles_proj(torch.mean(batch_Smiles, dim=1))
        ThreeD_embed = self.ThreeD_proj(torch.mean(batch_ThreeD, dim=1))
        Text_embed = self.Text_proj(batch_Text.last_hidden_state[:, 0, :])

        Smiles_feats = F.normalize(Smiles_embed, p=2, dim=-1)
        ThreeD_feats = F.normalize(ThreeD_embed, p=2, dim=-1)
        Text_feats = F.normalize(Text_embed, p=2, dim=-1)

        # loss for smiles_txt pair, threeD_txt pair, and smiles_threeD
        loss_smiles_txt = computeNCELoss(Smiles_feats, Text_feats, self.temperature)
        loss_threeD_txt = computeNCELoss(ThreeD_feats, Text_feats, self.temperature)
        loss_smiles_threeD = computeNCELoss(Smiles_feats, ThreeD_feats, self.temperature)

        return (loss_smiles_txt+loss_threeD_txt+loss_smiles_threeD)/3