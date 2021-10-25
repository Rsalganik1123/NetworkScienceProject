import json
import os
from tqdm import tqdm
import sys
import csv

class JSON_Loader(): 
    def __init__(self,): 
        self.playlist_fields = ['pid','name', 'collaborative', 'modified_at', 'num_albums', 'num_tracks', 'num_followers','num_tracks', 'num_edits', 'duration_ms', 'num_artists','description']

        self.track_fields = ['tid', 'arid' , 'alid', 'track_uri', 'track_name', 'duration_ms']

        self.album_fields = ['alid','album_uri','album_name']

        self.artist_fields = ['arid','artist_uri','artist_name']

        self.interaction_fields = ['pid','tid','pos']

        self.interactions = []

        self.playlists = []
        self.tracks = []
        self.artists = []
        self.albums = []

        self.count_files = 0
        self.count_playlists = 0
        self.count_interactions = 0
        self.count_tracks = 0
        self.count_artists = 0
        self.count_albums = 0
        self.dict_tracks = {}
        self.dict_artists = {}
        self.dict_albums = {}        
    def process(self, path, filenames): 
        for filename in sorted(filenames):
            if filename.startswith("mpd.slice.") and filename.endswith(".json"):
                fullpath = os.sep.join((path, filename))
                print("***Loading data from: {} ***".format(filename))
                f = open(fullpath)
                js = f.read()
                f.close()
                mpd_slice = json.loads(js)
                self.process_info(mpd_slice['info'])
                for playlist in mpd_slice['playlists']:
                    self.process_playlist(playlist)
                    pid = playlist['pid']
                    for track in playlist['tracks']:
                        track['pid']=pid
                        new = self.add_id_artist(track)
                        if new: self.process_artist(track)
                        new = self.add_id_album(track)
                        if new: self.process_album(track)
                        new = self.add_id_track(track)
                        if new: self.process_track(track)
                        self.process_interaction(track)
                    self.count_playlists += 1
                self.count_files +=1
            self.show_summary()
    def process_info(self,value):
        #print (json.dumps(value, indent=3, sort_keys=False))
        pass
    def add_id_track(self,track):
        if track['track_uri'] not in self.dict_tracks:
            self.dict_tracks[track['track_uri']] = self.count_tracks
            track['tid'] = self.count_tracks
            self.count_tracks += 1
            return True
        else:
            track['tid'] = self.dict_tracks[track['track_uri']]
            return False
    def add_id_artist(self, track):

        if track['artist_uri'] not in self.dict_artists:
            self.dict_artists[track['artist_uri']] = self.count_artists
            track['arid'] = self.count_artists
            self.count_artists += 1
            return True
        else:
            track['arid'] = self.dict_artists[track['artist_uri']]
            return False
        print("HERE",self.dict_artists)
    def add_id_album(self,track):
        # global count_albums
        if track['album_uri'] not in self.dict_albums:
            self.dict_albums[track['album_uri']] = self.count_albums
            track['alid'] = self.count_albums
            self.count_albums += 1
            return True
        else:
            track['alid'] = self.dict_albums[track['album_uri']]
            return False
    def process_track(self,track):
        # global track_fields
        info = []
        for field in self.track_fields:
            info.append(track[field])
        self.tracks.append(info)
    def process_album(self,track):
        # global album_fields
        info = []
        for field in self.album_fields:
            info.append(track[field])
        self.albums.append(info)
    def process_artist(self,track):
        # global artist_fields
        info = []
        for field in self.artist_fields:
            info.append(track[field])
        self.artists.append(info)
    def process_interaction(self,track):
        # global interaction_fields
        # global count_interactions
        info = []
        for field in self.interaction_fields:
            info.append(track[field])
        self.interactions.append(info)
        self.count_interactions +=1
    def process_playlist(self, playlist):
        # global playlist_fields
        if not 'description' in playlist:
            playlist['description'] = None
        info = []
        for field in self.playlist_fields:
            info.append(playlist[field])
        self.playlists.append(info)
    def show_summary(self):
        print("**** Loaded: {} files ****".format(self.count_files)) 
        print("**** Loaded: {} playlists ****".format(self.count_playlists))
        print("**** Loaded: {} tracks ****".format(self.count_tracks))
        print("**** Loaded: {} artists ****".format(self.count_artists))
        print("**** Loaded: {} albums ****".format(self.count_albums))
    def writer(self, path_save, init): 
        with open(path_save+"artists.csv", "a+") as f:
            writer = csv.writer(f,delimiter = "\t",)
            if init: 
                writer.writerow(self.artist_fields)
            writer.writerows(self.artists)
            print ("artists.csv done")

        with open(path_save+"albums.csv", "a+") as f:
            writer = csv.writer(f,delimiter = "\t",)
            if init: 
                writer.writerow(self.album_fields)
            writer.writerows(self.albums)
        print ("albums.csv done")
            
        with open(path_save+"train_interactions.csv", "a+") as f:
            writer = csv.writer(f,delimiter = "\t",)
            if init: 
                writer.writerow(self.interaction_fields)
            writer.writerows(self.interactions)
        print ("train_interactions.csv done")

        with open(path_save+"tracks.csv", "a+") as f:
            writer = csv.writer(f,delimiter = "\t",)
            if init: 
                writer.writerow(self.track_fields)
            writer.writerows(self.tracks)
        print ("tracks.csv done")

        with open(path_save+"train_playlists.csv", "a+") as f:
            writer = csv.writer(f,delimiter = "\t",)
            if init: 
                writer.writerow(self.playlist_fields)
            writer.writerows(self.playlists)
        print ("train_playlists.csv done")

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def batches(path): 
    filenames = [f for f in sorted(os.listdir(path)) if (f.startswith("mpd.slice.") and f.endswith(".json"))]
    batches = list(chunks(filenames, 10))
    init = True 
    for batch in batches:
        l = JSON_Loader()
        l.process(path, filenames=batch)
        l.writer("/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/", init)
        init = False

    # l = JSON_Loader()
    # l.process(path="/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_million_playlist_dataset/data/", filenames=["mpd.slice.0-999.json"])
    # l.writer("/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_in_csv/", init=True)

batches("/Users/rebeccasalganik/Documents/School/2021-2022/Network Science/Capstone/spotify_million_playlist_dataset/data/")