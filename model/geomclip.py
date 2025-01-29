"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip_models.blip_outputs import BlipOutput
from lavis.common.dist_utils import is_dist_avail_and_initialized
from model.geomclip_init import GeomCLIPBase
# from model.dist_funs import pl_concat_all_gather
from pytorch_lightning.utilities import distributed
# from torch_geometric.loader.dataloader import Collater
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    print('running here')
    return output

@torch.no_grad()
def pl_concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = distributed.gather_all_tensors(tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output


class GeomCLIP(GeomCLIPBase):
    def __init__(
        self,
        gtm,
        lm,
        bert_name,
        temperature,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        args=None,
        w_gt=0.5,
        w_ct=0.5,
    ):
        super().__init__()
        self.gtm = gtm
        self.lm = lm
        self.args = args
        self.tokenizer = self.init_tokenizer()
        # loss weight of diffrent contrastive modal
        self.w_gt = w_gt
        self.w_ct = w_ct
    
        self.conf_encoder, self.ln_conf, self.dictionary_mol = self.init_unimol_mol_encoder(args)

        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            #self.graph_encoder.train = disabled_train
            self.graph_encoder.train = False
            logging.info("freeze graph encoder")
        # for name, param in self.conf_encoder.named_parameters():
        #     param.requires_grad = False
        ### self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        self.Qformer = self.init_Qformer(bert_name, num_query_token, gin_hidden_dim, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
    
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.conf_proj = nn.Linear(args.unimol_encoder_embed_dim, embed_dim)
        # self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        # self.ctm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temperature_gt = temperature
        self.temperature_ct = temperature

    def contrast(self, features_graph, features_text, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        '''
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss
        
    def contrast_global_molclip(self, features_graph, features_text2d, feature_conf, feature_text3d, features_graph_all, \
                                features_text2d_all, feature_conf_all, feature_text3d_all, return_sim=False, w_gt=0.5, w_ct=0.5):
        
    
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)
        # cosine similarity as logits
        sim_g2t = torch.mm(features_graph, features_text2d_all.transpose(0, 1))
        # sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]
        logits_per_graph = sim_g2t / self.temperature_gt
        # sim_t2q = [(features_text.unsqueeze(1).unsqueeze(1) @ feature.permute(0, 2, 1)).squeeze() for feature in features_graph_all] # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        # sim_temp = []
        # for temp in sim_t2q:
        #     sim_t2g_t, _ = temp.max(-1)
        #     sim_temp.append(sim_t2g_t)
        # sim_t2g = torch.cat(sim_temp, dim=1)
        sim_t2g = torch.mm(features_text2d, features_graph_all.transpose(0, 1))
        logits_per_text2d = sim_t2g / self.temperature_gt

        #3D
        sim_c2t = torch.mm(feature_conf, feature_text3d_all.transpose(0, 1))
        logits_per_conf = sim_c2t / self.temperature_ct

        # sim_t2c = torch.mm(feature_text3d, feature_text3d_all.transpose(0, 1)) 
        sim_t2c = torch.mm(feature_text3d, feature_conf_all.transpose(0, 1))
        logits_per_text3d = sim_t2c / self.temperature_ct
        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = 0
        #rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text2d = F.cross_entropy(logits_per_text2d, labels)

        loss_conf = F.cross_entropy(logits_per_conf, labels)
        loss_text3d = F.cross_entropy(logits_per_text3d, labels)

        loss2D= (loss_graph + loss_text2d) / 2
        loss3D= (loss_conf + loss_text3d) / 2

        loss = self.w_gt*loss2D + self.w_ct*loss3D
        

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text2d[:, rank*bs:rank*bs+bs], \
                    logits_per_conf[:, rank*bs:rank*bs+bs], logits_per_text3d[:, rank*bs:rank*bs+bs], loss2D, loss3D,  loss
        else:
            return loss


    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze(dim=-1) # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    
        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze(dim=-2) # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
        else:
            return loss
        
    def forward(self, batch):
        conf_batch, text3d_tokens, text3d_mask = batch 
        batch_conf, batch_mask3d = self.conf_encoder(*conf_batch) # *tuple
 
        batch_conf = self.ln_conf(torch.mean(batch_conf, dim=1))#：shape = [B, n_max, D] [CLS] Token
        conf_feats = self.conf_proj(batch_conf)

        text3d_output = self.Qformer.bert(text3d_tokens, attention_mask=text3d_mask, return_dict=True) 
        text3d_feats = self.text_proj(text3d_output.last_hidden_state[:, 0, :])
      
        text3d_feats, conf_feats = F.normalize(text3d_feats, p=2, dim=-1), F.normalize(conf_feats, p=2, dim=-1)

        text3d_feats_all, conf_feats_all = pl_concat_all_gather(text3d_feats), pl_concat_all_gather(conf_feats)
        sim_g2t, sim_t2g, sim_c2t, sim_t2c, loss3D_, loss3D, loss_contra = self.contrast_global_molclip(conf_feats, text3d_feats, conf_feats, text3d_feats,\
                                            conf_feats_all, text3d_feats_all, conf_feats_all, text3d_feats_all, return_sim=True, w_gt=0.0, w_ct=1.0)
        return BlipOutput(
            loss=loss_contra,
            loss_itc=(loss3D_,loss3D),
            loss_itm=(loss3D_,loss3D)
        )
    
    
    # def graph_forward(self, graph):
    #     if self.args.use_3d:
    #         batch_node, batch_mask = self.graph_encoder(*graph)
    #     else:
    #         batch_node, batch_mask = self.graph_encoder(graph)
    #     batch_node = self.ln_graph(batch_node)
    #     query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
    #     query_output = self.Qformer.bert(
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=batch_node,
    #         encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
    #         use_cache=False,
    #         return_dict=True,
    #     )
    #     graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
    #     graph_feats = F.normalize(graph_feats, p=2, dim=-1)
    #     return graph_feats, batch_node, batch_mask

    # def text_forward(self, text, mask):
    #     text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
    #     text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
    #     text_feats = F.normalize(text_feats, dim=-1, p=2)
    #     return text_feats
    
    # def compute_gtm(self, batch_node, batch_mask, text_ids, text_atts):
    #     '''
    #     batch_node shape = [B, N, D]
    #     batch_mask shape = [B, N]
    #     text_ids shape = [B, N]
    #     text_atts shape = [B, N]
    #     '''
    #     query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1) # shape = [B, Nq, D]
    #     query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
    #         batch_node.device
    #     ) # shape = [B, Nq]
    #     attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
    #     output_gtm = self.Qformer.bert(
    #         text_ids,
    #         query_embeds=query_tokens,
    #         attention_mask=attention_mask,
    #         encoder_hidden_states=batch_node,
    #         encoder_attention_mask=batch_mask,
    #         return_dict=True,
    #     )
    #     gl_embeddings = output_gtm.last_hidden_state[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
    #     gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
    #     # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
    #     gtm_logit = gtm_logit[:, 1] # select the axis of the positive class
    #     return gtm_logit

