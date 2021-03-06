import os
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import csv
from tqdm import tqdm
from src.utils.utils import chunks

os.environ['SPOTIPY_CLIENT_ID'] = '81da2e7e661749f0a86024a8c2a04dd3'
os.environ['SPOTIPY_CLIENT_SECRET'] = '3d4324c79f2a441bb64c7a55d6a226fa'


class SPOTIFY_Loader():
    def __init__(self):
        self.music_feature_fields = ['track_uri', 'danceability', 'energy', 'key', 'loudness', 'speechiness',
                                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        self.feats = []

        self.count_tracks = 0
        self.skipped_tracks = 0

    def process(self, track_uri_list):
        sp = self.refresh_spotify()
        for uri in track_uri_list:
            try:
                track = sp.audio_features(tracks=[uri])
                self.process_music_features(uri, track[0])
                self.count_tracks += 1
            except Exception as e:
                print(e)
                self.skipped_tracks += 1
                time.sleep(10)
                self.refresh_spotify()
        self.show_summary()

    def process_music_features(self, track_uri, track):
        data = [track_uri]
        for field in self.music_feature_fields[1:]:
            data.append(track.get(field))
        self.feats.append(data)

    def show_summary(self):
        print("**** Loaded: {} tracks ****".format(self.count_tracks))
        print("**** Skipped {} tracks ****".format(self.skipped_tracks))

    def writer(self, path_save, init):
        with open(path_save + "music_features.csv", "a+") as f:
            writer = csv.writer(f, delimiter="\t", )
            if init:
                writer.writerow(self.music_feature_fields)
            writer.writerows(self.feats)
            self.feats = []
            print("feats.csv done")

    def refresh_spotify(self):
        auth_manager = SpotifyClientCredentials()
        sp = spotipy.Spotify(auth_manager=auth_manager)
        return sp


def load_uris(data_path):
    df = pd.read_csv(data_path, sep='\t')
    track_uri_list = list(df['track_uri'])
    # tid_list = list(df['tid'])
    uri_batches = list(chunks(track_uri_list, 100))
    # tid_batches = list(chunks(tid_list, 100))
    s = SPOTIFY_Loader()
    init = True
    for batch in tqdm(uri_batches):
        s.process(batch)
        s.writer('/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/', init)
        init = False


load_uris('/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/tracks.csv')

# self.danceability.append(x.get('danceability'))
#                     self.energy.append(x.get('energy'))
#                     self.key.append(x.get('key'))
#                     self.loudness.append(x.get('loudness'))
#                     self.speechiness.append(x.get('speechiness'))
#                     self.acousticness.append(x.get('acousticness'))
#                     self.instrumentalness.append(x.get('instrumentalness'))
#                     self.liveness.append(x.get('liveness'))
#                     self.valence.append(x.get('valence'))
#                     self.tempo.append(x.get('tempo'))
# data = {
#         'uri': x.get('uri'),
#         'danceability': x.get('danceability'),
#         'energy': x.get('energy'),
#         'key': x.get('key'),
#         'loudness': x.get('loudness'),
#         'speechiness': x.get('speechiness'),
#         'acousticness': x.get('acousticness'),
#         'instrumentalness': x.get('instrumentalness'),
#         'liveness': x.get('liveness'),
#         'valence': x.get('valence'),
#         'tempo': x.get('tempo')
#     }
