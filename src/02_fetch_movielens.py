import os
import zipfile
import requests

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_and_extract(dest='data/raw'):
    os.makedirs(dest, exist_ok=True)
    print("Downloading MovieLens...")
    r = requests.get(MOVIELENS_URL)
    zip_path = os.path.join(dest, 'ml-latest-small.zip')
    with open(zip_path, 'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    print("MovieLens dataset downloaded and extracted to", dest)

if __name__ == "__main__":
    download_and_extract()   