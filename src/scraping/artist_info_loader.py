import os
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import csv
from tqdm import tqdm
# from src.utils.utils import chunks

os.environ['SPOTIPY_CLIENT_ID'] = '81da2e7e661749f0a86024a8c2a04dd3'
os.environ['SPOTIPY_CLIENT_SECRET'] = '3d4324c79f2a441bb64c7a55d6a226fa'

class SPOTIFY_Loader():
    def __init__(self):
        self.artist_feature_fields = ['arid', 'artist_uri', 'followers', 'genres', 'popularity']
        self.feats = []

        self.count_tracks = 0
        self.skipped_tracks = 0

    def process(self, artist_uri_list):
        sp = self.refresh_spotify()
        for uri in tqdm(artist_uri_list):
            try:
                artist = sp.artist(uri)
                followers = artist['followers']['total']
                # print("Followers: {}, Genres: {}, Popularity: {}".format( followers, artist['genres'], artist['popularity']))
                self.process_artist_features(uri, artist, followers)
                self.count_tracks += 1
            except Exception as e:
                print(e)
                self.skipped_tracks += 1
                time.sleep(10)
                self.refresh_spotify()
        self.show_summary()

    def process_artist_features(self, artist_uri, artist, followers):
        data = [artist_uri, followers]
        for field in self.artist_feature_fields[3:]:
            data.append(artist.get(field))
        # print(data)
        self.feats.append(data)

    def show_summary(self):
        print("**** Loaded: {} tracks ****".format(self.count_tracks))
        print("**** Skipped {} tracks ****".format(self.skipped_tracks))

    def writer(self, path_save, init):
        with open(path_save + "artist_features2.csv", "a+") as f:
            writer = csv.writer(f, delimiter="\t", )
            if init:
                writer.writerow(self.artist_feature_fields)
            writer.writerows(self.feats)
            self.feats = []
            print("feats.csv done")

    def refresh_spotify(self):
        print("***Refreshing Spotify Token***")
        auth_manager = SpotifyClientCredentials()
        sp = spotipy.Spotify(auth_manager=auth_manager)
        return sp

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_uris(data_path):
    df = pd.read_csv(data_path, sep='\t')
    artist_uri_list = list(df['artist_uri'])
    # tid_list = list(df['tid'])
    uri_batches = list(chunks(artist_uri_list, 1000)) #300 
    # tid_batches = list(chunks(tid_list, 100))
    #I think it errors after 39 calls 
    s = SPOTIFY_Loader()

    init = False
    for batch in tqdm(uri_batches[161+12:200]): #hopefully 120 will be done # come back to this
        s.process(batch)
        s.writer('/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/', init)
        init = False

load_uris('/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/artists.csv')

# dic = {'followers': {'href': 0, "total": 9}}
# print(dic['followers', 'total'])