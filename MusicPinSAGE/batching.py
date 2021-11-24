from torch.utils.data import DataLoader, IterableDataset
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

class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]
