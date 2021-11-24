import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
from torch.nn.functional import cosine_similarity 

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

def select_topk(g, k, weight, nodes=None, edge_dir='in', ascending=False,
                copy_ndata=True, copy_edata=True):
    
    # Rectify nodes to a dictionary
    if nodes is None:
        nodes = {
            ntype: F.astype(F.arange(0, g.number_of_nodes(ntype)), g.idtype)
            for ntype in g.ntypes
        }
    elif not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    assert g.device == F.cpu(), "Graph must be on CPU."

    # Parse nodes into a list of NDArrays.
    nodes = utils.prepare_tensor_dict(g, nodes, 'nodes')
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
        else:
            nodes_all_types.append(nd.array([], ctx=nd.cpu()))

    if not isinstance(k, dict):
        k_array = [int(k)] * len(g.etypes)
    else:
        if len(k) != len(g.etypes):
            raise DGLError('K value must be specified for each edge type '
                           'if a dict is provided.')
        k_array = [None] * len(g.etypes)
        for etype, value in k.items():
            k_array[g.get_etype_id(etype)] = value
    k_array = F.to_dgl_nd(F.tensor(k_array, dtype=F.int64))

    weight_arrays = []
    for etype in g.canonical_etypes:
        if weight in g.edges[etype].data:
            weight_arrays.append(F.to_dgl_nd(g.edges[etype].data[weight]))
        else:
            raise DGLError('Edge weights "{}" do not exist for relation graph "{}".'.format(
                weight, etype))

    subgidx = _CAPI_DGLSampleNeighborsTopk(
        g._graph, nodes_all_types, k_array, edge_dir, weight_arrays, bool(ascending))
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)

    # handle features
    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, None)
        utils.set_new_frames(ret, node_frames=node_frames)

    if copy_edata:
        edge_frames = utils.extract_edge_subframes(g, induced_edges)
        utils.set_new_frames(ret, edge_frames=edge_frames)
    return ret

def prec(recommendations, ground_truth):
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))
    hit = relevance.any(axis=1).mean()
    return hit

class LatestNNRecommender(object):
    def __init__(self, user_ntype, item_ntype, user_to_item_etype, batch_size):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size

    def recommend(self, full_graph, K, h_user, h_item):
        """
        Return a (n_user, K) matrix of recommended items for each user
        """
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        n_users = full_graph.number_of_nodes(self.user_ntype)
        latest_interactions = select_topk(graph_slice, 1, 'weights', edge_dir='out')
        user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst')
        # each user should have at least one "latest" interaction
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(device=h_item.device)
            dist = h_item[latest_item_batch] @ h_item.t()
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(u, etype=self.user_to_item_etype)
                dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations

#for each playlist (with tracks) 
    #generate embedding of track 
    #Aggregate track embeddings --> playlist embedding 
    #dot product with all other tracks 
    #randomly sample 10000 and dot product those 

def evaluate_nn(dataset, h_item, k, batch_size):
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, batch_size)

    recommendations = rec_engine.recommend(g, k, 'contains', h_item).cpu().numpy() #was None, now 'contains'
    return prec(recommendations, val_matrix)

torch.seed()
def test_representations(model,  dataloader, device='cuda:0'): 
    with torch.no_grad():
            h_item_batches = []
            for blocks in tqdm.tqdm(dataloader):
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)
    return h_item

def score(g, model, test_track_dataloader,random_track_dataloader,  sample_num=1000):
    test_track_embs = test_representations(model, test_track_dataloader)
    print(test_track_embs.shape)
    playlist_emb = torch.mean(test_track_embs, dim=0).reshape((1, -1))
    #all_track_ids = torch.randint(1, g.number_of_nodes('track'), (sample_num,))
    option_embs = test_representations(model, random_track_dataloader)
    print(playlist_emb.shape)
    print(option_embs.shape)
    prods = torch.Tensor([cosine_similarity(playlist_emb, t) for t in option_embs])
    top = torch.topk(prods, 1)
    print(top) 
    return top


