
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import dgl
import os 
import sys 
import pickle 
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_categorical
import dgl.function as fn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score
import tqdm 
import scipy.sparse as ssp

from batching import * 
from neighbourhood import * 
from model import * 
from eval import * 

torch.seed()


def train(dataset, args):
    print("***DATASET LOADED***") 
    g = dataset['full-graph']
    train_g = dataset['train-graph']
    val_g = dataset['val-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']

    device = torch.device('cuda:0')
    
    print("Running on:{}".format(device))
    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes['playlist'].data['id'] = torch.arange(g.number_of_nodes('playlist'))
    g.nodes['track'].data['id'] = torch.arange(g.number_of_nodes('track'))

    #Add dummy weight to edges for evaluation step
    g.edges['contains'].data['weights'] = torch.ones(g.number_of_edges('contains'))

    # Sampler
    print("***SAMPLING***")
    batch_sampler = ItemToItemBatchSampler(g, 'playlist', 'track', 32) #was g
    neighbor_sampler = NeighborSampler(g, 'playlist', 'track', 
                                    random_walk_length=2, random_walk_restart_prob=0.5, num_random_walks=10, num_neighbors=3, num_layers=2)
    collator = PinSAGECollator(neighbor_sampler,g, 'track')

    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=4) 

    test_plist_ids = torch.randint(1, g.number_of_nodes(user_ntype), (1,))
    test_track_ids = g.successors(v=test_plist_ids, etype="contains") 

    dataloader_test = DataLoader(
        #torch.arange(g.number_of_nodes(item_ntype)),
        #torch.randint(1, g.number_of_nodes(item_ntype), (4000,)),
        #torch.arange(val_track_ids, val_track_ids+val_g.number_of_nodes('track')),
        test_track_ids,
        batch_size=32,
        collate_fn=collator.collate_test,
        num_workers=4)
    
    dataloader_test_options = DataLoader(
        #torch.arange(g.number_of_nodes(item_ntype)),
        #torch.randint(1, g.number_of_nodes(item_ntype), (4000,)),
        #torch.arange(val_track_ids, val_track_ids+val_g.number_of_nodes('track')),
        torch.randint(1, g.number_of_nodes('track'), (1000,)),
        batch_size=32,
        collate_fn=collator.collate_test,
        num_workers=4)
#all_track_ids = torch.randint(1, g.number_of_nodes('track'), (sample_num,))
    #test_plist_id = torch.randint(1, g.number_of_nodes(user_ntype), (1,))
    #test_track_ids = g.successors(v = test_plist_id, etype='contains') 
    print("TRACK IDS:", test_track_ids)
    

    #Model
    model = PinSAGEModel(g, 'track', 128, 2, args.emb)
    model = model.cuda()

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataloader_it = iter(dataloader)

    num_epochs = 10 
    batches_per_epoch = 10 
    k = 2
    batch_size = 32 
    losses = []
    
    print("***STARTING TRAIN***") 
    # For each batch of head-tail-negative triplets...
    for epoch_id in range(num_epochs):
        model.train()
        print ("EPOCH: {}".format(epoch_id))
        for batch_id in tqdm.trange(batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss, auc = model(pos_graph, neg_graph, blocks)
            loss = loss.mean() 
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch_id % 10 == 0:
            print(loss, auc)
            losses.append([loss.item(),auc])
         
        # Evaluate
        print("****EVALUATION START****")
        s = score(g, model, dataloader_test, dataloader_test_options) 
        print("!!!!!SCORE IS: {}!!!!!!".format(s)) 
        '''
        model.eval()
        with torch.no_grad():
            #ndcg = custom_score(test_track_ids, model)
            
    #         item_batches = #torch.arange(g.number_of_nodes(item_ntype)).split(batch_size)
            
            h_item_batches = []
            for blocks in tqdm.tqdm(dataloader_test):
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)
            s = score(g,model, h_item, sample_num=1000)
            print("****RECOMMENDED:{}****".format(s))
            #print(evaluate_nn(dataset, h_item, k, batch_size))
        '''    
        
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('dataset_path', type=str)
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--emb', type=bool, default=False)
    # parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    # parser.add_argument('--num-random-walks', type=int, default=10)
    # parser.add_argument('--num-neighbors', type=int, default=3)
    # parser.add_argument('--num-layers', type=int, default=2)
    # parser.add_argument('--hidden-dims', type=int, default=16)
    # parser.add_argument('--batch-size', type=int, default=32)
    # parser.add_argument('--device', type=str, default='cpu')        # can also be "cuda:0"
    # parser.add_argument('--num-epochs', type=int, default=1)
    # parser.add_argument('--batches-per-epoch', type=int, default=20000)
    # parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()

    directory='/home/mila/r/rebecca.salganik/'
    print(args)

     # Load dataset
    with open(directory+'dataset_without_im.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train(dataset, args)
