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

def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block

class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                                        random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    # print(old_frontier)
                    # print(frontier)
                    # print(frontier.edata['weights'])
                    # frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

def assign_features_to_blocks(blocks, g, ntype='track'):
    
    data = blocks[0].srcdata
    
    for col in g.nodes[ntype].data.keys():
        if  col == dgl.NID:
            continue
        induced_nodes = data[dgl.NID]
        data[col] = g.nodes[ntype].data[col][induced_nodes]

    
    data = blocks[-1].dstdata
    for col in g.nodes[ntype].data.keys():
        if  col == dgl.NID:
            continue
        induced_nodes = data[dgl.NID]
        data[col] = g.nodes[ntype].data[col][induced_nodes]

class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g
        #self.g_test = g_test

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.ntype)

        return pos_graph, neg_graph, blocks
    
    def collate_test(self, samples):
        batch = torch.LongTensor(samples) 
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.ntype)
        return blocks        
