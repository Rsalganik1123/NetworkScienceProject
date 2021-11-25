import pickle
import argparse
import numpy as np
import torch as th
import torch
import torch.nn as nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader, IterableDataset
import dgl
import scipy.sparse as ssp
import os
import sys
import pickle
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_categorical
import dgl.function as fn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score
import tqdm
from sklearn.linear_model import LogisticRegression
from dgl.sampling import node2vec_random_walk
from node2vec import Node2vecModel
from mlp import MLP
directory = '/Users/juju/pinsage_cluster/'
device = th.device('cuda:0')
#get data
def get_data():

    with open( directory + 'dataset_without_im_copy.pkl', 'rb') as f:
        dataset = pickle.load(f)

    g = dataset['full-graph']

    #g = dgl.to_homogeneous(g) #not quite yet afterall
    return g



#Old way of building training
def old_way_to_get_training_playlists(g):
    #need to get random pairs of playlist_item and track_item that do have an edge together
    TRAINING_SIZE = 500
    TESTING_SIZE = 100
    max_id = len(g.edges(etype='contains')[0])
    np.random.seed(42)
    edge_pos_id = np.random.randint(max_id, size=(TRAINING_SIZE//2)+(TESTING_SIZE//2))
    #edge_test_pos_id = np.random.randint(max_id, size=TESTING_SIZE//2)


    #wait this doesn't work, because now edges both for contain and contained by??
    source = g.edges(etype='contains')[0]
    destination = g.edges(etype='contains')[1]
    #print("source", source)
    def th_keep(tensor, indices):
        mask = torch.zeros(len(tensor), dtype=th.bool)
        mask[indices] = True
        return tensor[mask]
    #print(len(g.edges()[0]))
    print(g.edges(etype='contains', form='eid'))
    ids_to_remove = th_keep(g.edges(etype='contains', form='eid'), edge_pos_id)
    g.remove_edges(ids_to_remove, etype='contains')
    #print(len(g.edges()[0]))
    # a= th_keep(source, edge_train_pos_id)
    # b = th_keep(destination, edge_train_pos_id)
    # print(a)
    # print(b)
    pos_edges = torch.stack([th_keep(source, edge_pos_id), th_keep(destination, edge_pos_id)])

    #now get negative edges, just do true random, and because graph is so big, we can believe we will get
    source_id = np.random.randint(max_id, size=(TRAINING_SIZE//2)+(TESTING_SIZE//2))
    dest_id = np.random.randint(max_id, size=(TRAINING_SIZE//2)+(TESTING_SIZE//2))

    neg_edges = torch.stack([th_keep(source, source_id), th_keep(destination, dest_id)])

    #g.edges()[1] = th_delete(destination, edge_train_pos_id)
    #g.edata['_ID'] = th_delete(temp_data['_ID'], edge_train_pos_id)
    #g.edata['_TYPE'] = th_delete(temp_data['_TYPE'], edge_train_pos_id)

    return g, pos_edges, neg_edges

def train_embeddings(g):
    model = Node2vecModel(g, embedding_dim=64, walk_length=10, p=1, q=1)
    print("model initiated")
    model.train(epochs=10, batch_size=20)
    embeddings = model.embedding()
    print(embeddings)


    with open(directory + 'embeddings_10_epochs.pkl', 'wb') as f:
        pickle.dump(embeddings, f)


class RecDataset(torch.utils.data.Dataset):
    def __init__(self, pos_edges, neg_edges):
        # pos_edges.shape = (2,test_size+train_size/2)
        # neg_edges.shape = (2,test_size+train_size/2)
        # initializing the data
        pos = pos_edges.t()
        neg = neg_edges.t()

        # inputs = torch.cat([embeddings[inputs[0]],embeddings[inputs[1]])
        self.x = torch.cat([pos, neg])

        #         y_pos = torch.stack([torch.ones(pos.size()[0]), torch.zeros(pos.size()[0])]).t()
        #         y_neg = torch.stack([torch.zeros(neg.size()[0]), torch.ones(neg.size()[0])]).t()
        # NEVERMIND: y shape=(2,), [1,0] if link does exists, [0,1] if it doesn't
        #shape of y is (1,) #binary output 1 or 0
        # self.y = torch.cat([y_pos,y_neg])
        self.y = torch.cat([torch.ones(pos.size()[0]), torch.zeros(neg.size()[0])])
        self.n_samples = len(pos_edges[0]) + len(neg_edges[0])

    def __getitem__(self, index):
        # dataset[]
        return (self.x[index], self.y[index])

    def __len__(self):
        # return len(dataset)
        return self.n_samples


def training_loop(mlp, embeddings, dataloader, optimizer, loss_function):
    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')
        # Set current loss value
        current_loss = 0.0
        #set current accuracy value
        accuracy = 0.0
        # Iterate over the DataLoader for training data
        num = 1
        for i, data in enumerate(dataloader, 0):

            # Get inputs
            inputs_no_embedding, targets = data

            targets = targets.unsqueeze(1)
            # targets = torch.tensor(targets, dtype=torch.long)

            inputs = []
            for j in inputs_no_embedding:
                inputs.append(torch.cat([embeddings[j[0]], embeddings[j[1]]]))
            # print("inputs: ",inputs)
            inputs = torch.stack(inputs)
            #             print(inputs)
            #             return
            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            def get_accuracy(outputs, targets):
                acc = 0
                count = len(outputs)
                for datapoint in range(len(outputs)):
                    y_hat= outputs[datapoint]
                    y = targets[datapoint]
                    if y == 0 and y_hat<0.5:
                        acc=+1
                    elif y==1 and y_hat>=0.5:
                        acc=+1
                return acc/count

            accuracy =+ get_accuracy(outputs, targets)
            # Compute loss
            #print("ouputs: ", outputs)
            #print("targers: ", targets)
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            num = num + 1

        print('epoch %5d, average loss : %.3f, average accuracy : %.3f' %
        (epoch, current_loss/num, accuracy/num))


    # Process is complete.
    print('Training process has finished.')

#let's train the MLP
def main():
    #open all data
    #not done yet, just pushing to keep this version for when i do changes
    # all_data = pickle.load(open('ns_music_all_data_ming_20211124', 'rb'))
    # #dict_keys(['df_playlist_info', 'df_playlist', 'df_track', 'train_indices',
    # # 'val_indices', 'test_indices', 'train_user_ids', 'val_user_ids', 'test_user_ids'])
    # df_playlist = all_data['df_playlist']
    # df_playlist = df_playlist.sort_values(['pid', 'pos'])
    # train_user_ids = all_data['train_user_ids']
    # val_user_ids = all_data['val_user_ids']
    # test_user_ids = all_data['test_user_ids']
    #
    # train_indices = all_data['train_indices']

    g, train_indices, etype, etype_rev = pickle.load(open('/Users/juju/Downloads/g_train_etype_etyperev.pkl', 'rb'))
    train_g = g.edge_subgraph({etype: train_indices, etype_rev: train_indices})
    embeddings = train_embeddings(train_g)
    print(train_g)
    return
    g= get_data()
    g, pos_edges, neg_edges = old_way_to_get_training_playlists(g)
    g = dgl.to_homogeneous(g)
    with open(directory + 'emb3epoch.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    #if you need to train them
    #embeddings = train_embeddings(g)
    recData = RecDataset(pos_edges, neg_edges)
    inputs, targets = recData[0]
    print(embeddings[inputs[0]])
    dataloader = DataLoader(recData, batch_size=5, shuffle=True)
    # Initialize the MLP
    mlp = MLP()
    # Define the loss function and optimizer
    loss_function = BCELoss()  # binary cross entropy loss
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    training_loop(mlp, embeddings, dataloader, optimizer, loss_function )


main()