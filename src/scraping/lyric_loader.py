import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
import requests
import csv
import time
import re
from tqdm import tqdm
import os
from lyricsgenius import Genius
from requests.exceptions import HTTPError, Timeout

token = 'XxjduohJNSubKbqL47-dEAO6nSDtWgawot7hwF5qyyemBKT7yb0EwIOiqHt84SSC'
genius = Genius(token)


class SongNotInDB(Exception):
    def __init__(self, message="Song not found in Genius DB"):
        self.message = message
        super().__init__(self.message)


class NotLyrical(Exception):
    def __init__(self, message="Song doesn't have lyrics"):
        self.message = message
        super().__init__(self.message)


class Lyric_Loader():
    def __init__(self):
        self.track_count = 0
        self.skipped_tracks = 0
        self.lyric_fields = {"arid", "track", "artist", "lyrics"}
        self.lyrics = []
        self.skipped_value_names = []

    def process(self, batch):
        for arid, track, artist in batch:
            try:
                track, artist = self.clean_text(track, artist)
                song = genius.search_song(track, artist)
                if not song: raise SongNotInDB()
                lyrics = song.lyrics
                if not lyrics:
                    artist = genius.search_artist(artist, max_songs=1, sort="title")
                    lyrics = artist.song(track)
                if not lyrics: raise NotLyrical()
                lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
                regex = re.compile('[^a-zA-Z \n]')
                lyrics = regex.sub('', lyrics)
                # lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
                self.store_lyrics(arid, track, artist, lyrics)
                self.track_count += 1
            except Exception as e:
                print("E: {}, A: {}, T: {}".format(e, artist, track))
                time.sleep(5)
                self.skipped_value_names.append((artist, track))
                self.skipped_tracks += 1
        self.show_summary()

    def clean_text(self, track, artist):
        track, artist = track.split("(")[0], artist.split("(")
        track = track.split("-")[0]
        track = track.lower()
        return track, artist

    def store_lyrics(self, arid, track, artist, lyrics):
        data = [arid, track, artist, lyrics]
        self.lyrics.append(data)

    def writer(self, path_save, init):
        with open(path_save + "lyrics.csv", "a+") as f:
            writer = csv.writer(f, delimiter="\t", )
            if init:
                writer.writerow(self.lyric_fields)
            writer.writerows(self.lyrics)
            self.feats = []
            print("lyrics.csv done")
        with open(path_save + "missed_lyric_scape_vals.txt", "a") as f:
            for v in self.skipped_value_names:
                f.write(' '.join(str(s) for s in v) + '\n')

    def show_summary(self):
        print("**** Loaded: {} tracks ****".format(self.track_count))
        print("**** Skipped {} tracks ****".format(self.skipped_tracks))


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_batches(track_path, artist_path):
    track_df = pd.read_csv(track_path, sep='\t')
    artist_df = pd.read_csv(artist_path, sep='\t')
    joined_df = track_df.join(artist_df.set_index('arid'), on='arid')
    track = list(joined_df['track_name'])
    arid = list(joined_df['arid'])
    artist = list(joined_df['artist_name'])
    data = list(zip(arid, track, artist))
    batches = list(chunks(data, 10))
    l = Lyric_Loader()
    init = True
    for batch in tqdm(batches):
        l.process(batch)
        l.writer('/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/', init)
        init = False

    # track_path, artist_path = "/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/tracks.csv", "/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/artists.csv"


# load_batches(track_path, artist_path)


# track_df = pd.read_csv(track_path, sep='\t').reset_index(drop=True)
# artist_df = pd.read_csv(artist_path, sep='\t').reset_index(drop=True)
# track = list(track_df['track_name'])
# arid = list(track_df['arid'])
# artist = list(artist_df[artist_df['arid'].isin(arid)]['artist_name'])
# data = list(zip(arid, artist, track))
# l = Lyric_Loader()
# url = 'https://genius.com/missy-elliott-lose-control-lyrics'
# print(l.scrape_song_lyrics(url))
# artist, track = l.preprocess_before_lyric_scrape(artist[0], track[0])
# print(l.scrape_lyrics(artist, track))

try:
    track, artist = 'danse macabre', 'Camille Saint-Saëns'
    song = genius.search_song(track, artist)
except Exception as e:
    print(e)
    print("MESSAGE", e.errno)  # status code
    print(e.args[0])  # status code
    print(e.args[1])  # error message
# except HTTPError as e:
#     print(e.errno)    # status code
#     print(e.args[0])  # status code
#     print(e.args[1])  # error message
# except Timeout:
#     pass
