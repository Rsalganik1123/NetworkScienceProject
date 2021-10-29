import os

os.environ['CUDA_VISIBLE_DEVICES'] = '9'
import time
import pickle
import pandas as pd
from lyricsgenius import Genius
from IPython.utils import io
import numpy as np
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def clean_text(artist, track):
    track, artist = track.split("(")[0], artist.split("(")[0]
    track = track.split("-")[0]
    track = track.split('feat.')[0]
    track = track.lower()

    return artist, track


def create_connector():
    token = 'XxjduohJNSubKbqL47-dEAO6nSDtWgawot7hwF5qyyemBKT7yb0EwIOiqHt84SSC'
    genius = Genius(token)
    return genius


def load_embed_lyrics(data_list):
    genius = create_connector()

    output_data = []
    with io.capture_output() as captured:

        for entry in data_list:
            artist, track = clean_text(entry['artist_name'], entry['track_name'])

            data = {
                'track_uri': entry['track_uri'],
                'track_name': entry['track_name'],
                'artist_name': entry['artist_name']
            }

            try:
                song = genius.search_song(track, artist)
                lyrics = song.lyrics
                scraped_artist = song.artist
                scraped_track = song.title
                data['lyrics'] = lyrics
                data['scraped_track_name'] = scraped_track
                data['scraped_artist_name'] = scraped_artist
            except:
                pass

            output_data.append(data)
    return output_data


def compute_lyrics_emb(model, lyrics):
    sentences = lyrics.split('\n')
    sentences = [x for x in sentences if len(x) > 0]

    lyrics_emb = []
    for b in chunks(sentences, 32):
        lyrics_emb.append(model.encode(b))
    lyrics_emb = np.vstack(lyrics_emb)
    lyrics_emb = np.mean(lyrics_emb, axis=0)

    return lyrics_emb

track_path = '/home/justine/Documents/recsys/spotify_in_csv/tracks.csv'
artist_path = '/home/justine/Documents/recsys/spotify_in_csv/artists.csv'
folder = '/home/justine/Documents/recsys/results'

track_df = pd.read_csv(track_path, sep='\t')
track_uris = sorted(list(track_df['track_uri']))
track_uri_chunk = track_uris[754097*2:]
track_df = track_df[track_df['track_uri'].isin(track_uri_chunk)]
artist_df = pd.read_csv(artist_path, sep='\t')
joined_df = track_df.join(artist_df.set_index('arid'), on='arid')
data = joined_df [['track_uri', 'artist_name', 'track_name']].to_dict(orient='records')

large_batches = list(chunks(data, 1000))
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

for idx, large_batch in enumerate(large_batches):
    print('Processing batch ', idx)
    batches = list(chunks(large_batch, 100))
    p = Pool(10)
    out = p.map(load_embed_lyrics, batches)
    out = sum(out, [])
    p.terminate()

    # compute embeddings
    for batch in chunks(out, 32):

        titles = [x['track_name'] for x in batch]
        tracks_emb = model.encode(titles)
        for emb, x in zip(tracks_emb, batch):
            x['track_name_emb'] = emb
    for entry in out:
        if 'lyrics' in entry:
            lyrics_emb = compute_lyrics_emb(model, entry['lyrics'])
            entry['lyrics_emb'] = lyrics_emb

    path = os.path.join(folder, 'data_{0}.p'.format(idx))
    print('save batch {0} to path {1}'.format(idx, folder))
    pickle.dump(out, open(path, 'wb'))