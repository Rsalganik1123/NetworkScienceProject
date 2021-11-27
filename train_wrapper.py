from src.configs.defaults import get_cfg_defaults
import sys 
import os 

cfg = get_cfg_defaults()
#print(cfg)

# change data path
cfg.merge_from_file('/home/mila/r/rebecca.salganik/NetworkScienceProject/src/configs/spotify_recsys/music_genre_ids_imgemb_64_smalldata.yaml')
cfg.DATASET.DATA_PATH = '/home/mila/r/rebecca.salganik/ns_music_all_data_with_split'
#cfg_data.DATA_PATH = '/home/mila/r/rebecca.salganik/ns_music_all_data_with_split'
#cfg_data = cfg.DATASET

cfg.TRAIN.LOSS = 'FOCAL_LOSS'
cfg.TRAIN.EPOCHS=10
cfg.TRAIN.BATCHES_PER_EPOCH = 50000

# specify output path
#cfg.OUTPUT_PATH = '/home/mila/r/rebecca.salganik/scratch/'
cfg.OUTPUT_PATH = os.path.join('/home/mila/r/rebecca.salganik/scratch/', sys.argv[-1])

cfg.DATASET.TRAIN_INDICES = 'train_toptestval_indices'

'''
cfg.MODEL.PINSAGE.PROJECTION.NORMALIZE= True
cfg.MODEL.PINSAGE.REPRESENTATION_NORMALIZE= True
cfg.MODEL.PINSAGE.HIDDEN_SIZE = 64
cfg.MODEL.PINSAGE.PROJECTION.EMB = [['id', 64],
 ['album_id', 64],
 ['artist_id', 64],
 ['key', 16],
 ['tempo_5cat', 8],
 ['livness_5cat', 8],
 ['instrumentalness_3cat', 4],
 ['speechiness_10cat', 16],
 ['loudness_10cat', 16],
 ['acousticness_10cat', 16]]
#cfg.merge_from_file('music_genre_ids_imgembd_64_smalldata.yaml') 
'''
from src.train_net import train

train(cfg)
