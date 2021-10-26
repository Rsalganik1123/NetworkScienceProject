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

token = 'XxjduohJNSubKbqL47-dEAO6nSDtWgawot7hwF5qyyemBKT7yb0EwIOiqHt84SSC'
genius = Genius(token)

class Lyric_Loader(): 
    def __init__(self): 
        self.track_count = 0 
        self.skipped_tracks = 0 
        self.lyric_fields = {"tid", "lyrics"} 
        self.lyrics = [] 
        self.skipped_value_names = [] 
    def process2(self, batch): 
        for tid, artist, track in batch: 
            try: 
                song = genius.search_song(track, artist)
                lyrics = song.lyrics
                lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
                lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
                self.store_lyrics(tid, lyrics)
                self.track_count += 1 
            except Exception as e: 
                print("E: {}, A: {}, T: {}".format(e, artist, track)) 
                time.sleep(5)
                self.skipped_value_names.append((artist,track))
                self.skipped_tracks += 1
        self.show_summary()    
    def process(self, batch): 
        for tid, artist, track in batch: 
            try: 
                artist, track = self.clean_text(artist, track)
                request_string = self.prepare_request_string(artist, track)
                print(request_string)
                # lyrics = self.scrape_lyrics(request_string)
                lyrics = self.scrape_song_lyrics(request_string)
                print(lyrics[:10])
                # lyrics = self.preprocessing(lyrics)
                self.store_lyrics(tid, lyrics)
                self.track_count += 1 
            except Exception as e: 
                print("E: {}, A: {}, T: {}".format(e, artist, track)) 
                time.sleep(10)
                self.skipped_value_names.append((artist,track))
                self.skipped_tracks += 1
        self.show_summary()   
    def clean_text(self, artist, track): 
        # print("A: {}, T: {}".format(artist, track))
        artist, track = artist.lower(), track.lower()
        artist, track = artist.split("(")[0], track.split("(")[0]
        artist, track = artist.split(), track.split() 
        artist, track = [re.sub("[^a-zA-Z]+", "", a) for a in artist], [re.sub("[^a-zA-Z]+", "", t) for t in track]
        print("A: {}, T: {}".format(artist, track))
        return artist, track
    def prepare_request_string(self, artist, track): 
        request_string = 'https://genius.com/' + artist[0] #.capitalize()
        for a in artist[1:]: 
            request_string += "-" + a
        for t in track: 
            request_string += "-" + t 
        request_string += '-lyrics'
        return request_string   
    def scrape_song_lyrics(self, url):
        page = requests.get(url)
        html = BeautifulSoup(page.text, 'html.parser')
        
        lyrics = html.find('div', class_='lyrics')
        print(lyrics)
        lyrics = lyrics.get_text()
        #remove identifiers like chorus, verse, etc
        lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
        #remove empty lines
        lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])         
        return lyrics
    def scrape_lyrics(self, request_string):
        page = requests.get(request_string)
        html = BeautifulSoup(page.text, 'html.parser')
        lyrics = html.find('div', class_='lyrics').get_text()
        # print(lyrics1.get_text())
        # lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-2 jgQsqn")
        # print(lyrics2.get_text())
        # if lyrics1:
        #     lyrics = lyrics1.get_text()
        # elif lyrics2:
        #     lyrics = lyrics2.get_text()
        # elif lyrics1 == lyrics2 == None:
        #     lyrics = []
        # else: return None 
        return lyrics
    def preprocessing(self, lyrics): 
        return lyrics 
    def store_lyrics(self, tid, lyrics): 
        data = [tid, lyrics]
        self.lyrics.append(data)    
    def writer(self, path_save, init): 
        with open(path_save+"lyrics.csv", "a+") as f:
                writer = csv.writer(f,delimiter = "\t",)
                if init: 
                    writer.writerow(self.lyric_fields)
                writer.writerows(self.lyrics)
                self.feats = [] 
                print ("lyrics.csv done") 
        with open(path_save+"missed_lyric_scape_vals.txt", "a") as f:
            for v in self.skipped_value_names: 
                f.write(' '.join(str(s) for s in v) + '\n')
    def show_summary(self): 
        print("**** Loaded: {} tracks ****".format(self.track_count))
        print("**** Skipped {} tracks ****".format(self.skipped_tracks))  

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_batches(track_path, artist_path): 
        track_df = pd.read_csv(track_path, sep='\t').reset_index(drop=True)
        artist_df = pd.read_csv(artist_path, sep='\t').reset_index(drop=True)
        track = list(track_df['track_name'])
        arid = list(track_df['arid']) 
        tid = list(track_df['tid'])
        artist = list(artist_df[artist_df['arid'].isin(arid)]['artist_name']) 
        data = list(zip(tid, artist, track))
        batches = list(chunks(data, 10))
        l = Lyric_Loader()
        init = True
        for batch in tqdm(batches): 
            l.process2(batch)
            l.writer('/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/', init)
            init = False 
            break 

track_path, artist_path = "/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/tracks.csv", "/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/artists.csv" 
load_batches(track_path, artist_path)
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
