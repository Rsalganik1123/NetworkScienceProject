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
from src.utils.utils import chunks
token = 'XxjduohJNSubKbqL47-dEAO6nSDtWgawot7hwF5qyyemBKT7yb0EwIOiqHt84SSC'
genius = Genius(token)
import numpy 
from sentence_transformer import SentenceTransformer 

class WrongEntry(Exception): 
    def __init__(self, message="Wrong song entry found"):
        self.message = message
        super().__init__(self.message)

class Lyric_Loader():
    def __init__(self):
        self.track_count = 0
        self.skipped_tracks = 0
        self.lyric_fields = {"arid", "track", "artist","gen_t", "gen_a", "correct_lyric_scrape", "lyrics"}
        self.lyrics = []
        self.skipped_value_names = []
        self.potentially_wrong_entries = [] 
    def process(self, batch):
        for arid, track, artist in batch:
            try:
                track, artist = self.clean_text(track, artist)
                song = genius.search_song(track, artist)
                lyrics = song.lyrics
                _, gen_a = self.clean_text("", song.artist)
                correct_lyric_scrape = gen_a.lower() == artist.lower()
                print("GENIUS T: {}, GENIUS A: {}, CORRECT?: {}".format(song.title, gen_a, correct_lyric_scrape))
                if not correct_lyric_scrape: 
                    self.potentially_wrong_entries.append((arid, gen_t, gen_a))
                #     raise WrongEntry()
                # lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
                # regex = re.compile('[^a-zA-Z \n]')
                # lyrics = regex.sub('', lyrics)
                #Preprocess here (before you store) with: self.preprocess_lyrics(lyrics)
                self.store_lyrics(arid, track, artist, gen_t, gen_a, correct_lyric_scrape, lyrics)
                self.track_count += 1
            except Exception as e:
                print("E: {}, A: {}, T: {}".format(e, artist, track))
                time.sleep(5)
                self.skipped_value_names.append((artist, track))
                self.skipped_tracks += 1
        self.show_summary()

    def clean_text(self, track, artist):
        track, artist = track.split("(")[0], artist.split("(")[0]
        track = track.split("-")[0]
        track = track.lower()
        return track, artist

    def store_lyrics(self, arid, track, artist, gen_t, gen_a, correct_lyric_scrape, lyrics):
        data = [arid, track, artist, gen_t, gen_a, correct_lyric_scrape, lyrics]
        self.lyrics.append(data)

    def preprocess_lyrics(self, lyrics): 
        lyrics = lyrics.split('\n')
        average = numpy.array([0] * 512)
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        counter = 0
        for i in lyrics:
            if i != '':
                average = average + numpy.array(model.encode(i))
                counter = counter + 1

        emb = average / counter
        return emb 
    
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
        with open(path_save + "potentially_wrong_lyrics.txt", "a") as f:
            for v in self.potentially_wrong_entries:
                f.write(' '.join(str(s) for s in v) + '\n')

    def show_summary(self):
        print("**** Loaded: {} tracks ****".format(self.track_count))
        print("**** Skipped {} tracks ****".format(self.skipped_tracks))

def load_batches(track_path, artist_path, write_path):
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
        # l.writer(write_path, init)
        init = False

path = '/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/'
track_path, artist_path, write_path = path +"/spotify_in_csv/tracks.csv", path + "/spotify_in_csv/artists.csv", path + "spotify_in_csv/"

# load_batches(track_path, artist_path, write_path)


def tester(): 
    l = Lyric_Loader()
    tid, track, artist = 1, 'Symphony No. 7', 'Ludwig Van Beethoven'
    batch = [(tid, track, artist)]
    l.process(batch)

    try:
        track, artist = 'Symphony No. 7', 'Ludwig Van Beethoven'
        artist = genius.search_artist(artist, max_songs=1, sort="title")
        print(artist)
        lyrics = artist.song(track).lyrics
        print(lyrics[:50])
        
        # song = genius.search_song(track, artist)

    except Exception as e:
        print(e)
        print("MESSAGE", e.errno)  # status code
        print(e.args[0])  # status code
        print(e.args[1])  # error message

