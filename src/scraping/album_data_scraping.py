import os
import pickle
import requests
from tqdm import tqdm
from src.utils.utils import chunks
from src.utils.spotify_connector import refresh_spotify, load_album_data


def scrape_albums_data(albums_uri, imgs_folder, meta_path):
    """
    given list of album uris, load and save album data, save images as well
    :param albums_uri:  list of spotify album uris
    :param imgs_folder: output images folder
    :param meta_path:   output meta path
    :return: metadata of scraped data, missing uris
    """
    all_data = []
    batches = list(chunks(albums_uri, 10))
    missed_albums = []
    sp = refresh_spotify()
    for batch in tqdm(batches):
        try:
            results = load_album_data(sp, batch)
        except:
            missed_albums = missed_albums + batch
        else:
            for uri, album in zip(batch, results['albums']):
                try:
                    data = {
                        'uri': album['uri'],
                        'name': album['name'],
                        'popularity': album['popularity'],
                        'release_date': album['release_date'],
                        'total_tracks': album['total_tracks']
                    }
                    imgs = album['images']
                    if len(imgs) > 0:
                        if len(imgs) > 1:
                            idx = 1
                        else:
                            idx = 0
                        img_url = album['images'][idx]['url']
                        img_data = requests.get(img_url).content
                        dl_path = os.path.join(imgs_folder, '{0}.jpg'.format(album['uri']))
                        with open(dl_path, 'wb') as handler:
                            handler.write(img_data)
                        data['image_path'] = dl_path
                    else:
                        data['image_path'] = 'NO_IMAGE'
                    all_data.append(data)
                except:
                    if album is not None:
                        raise
                    missed_albums.append(uri)
    pickle.dump(all_data, open(meta_path, 'wb'))
    return all_data, missed_albums
