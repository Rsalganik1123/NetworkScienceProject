"""
All functionalities related to spotify data loading
"""

import os
import time
import json
import spotipy
import requests
import pandas as pd
from tqdm import tqdm
from spotipy.oauth2 import SpotifyClientCredentials

# TODO CLEAN THIS
os.environ['SPOTIPY_CLIENT_ID'] = '81da2e7e661749f0a86024a8c2a04dd3'
os.environ['SPOTIPY_CLIENT_SECRET'] = '3d4324c79f2a441bb64c7a55d6a226fa'


def refresh_spotify():
    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp


def load_album_data(sp, batch, retry_num=10):
    if retry_num < 0:
        raise Exception('Error')

    retry_num -= 1
    try:
        results = sp.albums(batch)
        return results
    except Exception as e:
        print(e)
        sp = refresh_spotify()
        time.sleep(10)

    return load_album_data(sp, batch, retry_num)
