import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_categorical
import torch
import pandas as pd
import dgl
import dgl.function as fn
from sklearn.model_selection import train_test_split

import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader


from torch import nn
from sklearn.metrics import roc_auc_score, ndcg_score
import tqdm
import scipy.sparse as ssp

def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False


def _init_input_modules(g, ntype):
    module_dict = nn.ModuleDict()
    
    tracks_data = g.nodes[ntype].data
    
    module_dict['track_id'] = nn.Embedding(tracks_data['id'].max()+1, 128)
    
    for m in ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']: 
        module_dict[m] = nn.Embedding(tracks_data[m].max() + 1, 16)

    return module_dict


class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """

    def __init__(self, full_graph, ntype, emb):
        super().__init__()

        self.ntype = ntype
        self.fc = nn.Linear(164, 128)
        self.emb = emb
        #self.fc = nn.Linear(2212, 128)
        self.inputs = _init_input_modules(full_graph, ntype)

    def forward(self, ndata):
        
        # get music feature
        music_features = []
        for c in ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']:

            module = self.inputs[c]
            music_features.append(module(ndata[c]))
        music_features = torch.cat(music_features, dim=1)
        
        
        # id embedding 
        id_embedding = self.inputs['track_id'](ndata['id'])
        
        # album feature 
        if self.emb: 
            img_emb = ndata['album_img_emb']
        
        # genre 
        genre = ndata['genre']
        
        # concatenate 
        if self.emb: 
            feature = torch.cat([music_features, genre, ndata['album_img_emb']], dim=1)
        feature = torch.cat([music_features, genre], dim=1)

        projection = self.fc(feature) + id_embedding
        
        return projection

class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z


class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        """
        g : DGLHeteroGraph
            The user-item interaction graph.
            This is only for finding the range of categorical variables.
        item_textsets : torchtext.data.Dataset
            The textual features of each item node.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h

class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()

        n_nodes = full_graph.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.edata['s'] = item_item_graph.edata['s'].flatten()
            item_item_graph.apply_edges(self._add_bias)

            pair_score = item_item_graph.edata['s']
        return pair_score

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, hidden_dims, n_layers, emb):
        super().__init__()

        self.proj = LinearProjector(full_graph, ntype, emb)
        self.sage = SAGENet(hidden_dims, n_layers)
        self.scorer = ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        
        #return h_item, pos_score, neg_score
        auc = compute_auc(pos_score, neg_score)
        return (neg_score - pos_score + 1).clamp(min=0), auc 


    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)
