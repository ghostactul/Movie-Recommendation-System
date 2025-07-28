import requests
import json
import time
import os
import pandas as pd

# === Config ===
API_KEY = '96fd05ded6df51981aabef76099b9b3f'  # Replace this
BASE_URL = 'https://api.themoviedb.org/3'

MOVIE_FILE = 'tmdb_movies.json'
LANGUAGE_FILE = 'tmdb_languages.json'
RATINGS_FILE = './ml-32m/ratings.csv'  # Your real MovieCine rating data
MERGED_FILE = 'merged_moviecine_tmdb.csv'

# === 1. Load or fetch TMDb language mappings ===
def fetch_language_mapping():
    url = f"{BASE_URL}/configuration/languages"
    params = {'api_key': API_KEY}
    response = requests.get(url, params=params)
    return {lang['iso_639_1']: lang['english_name'] for lang in response.json()}

if not os.path.exists(LANGUAGE_FILE):
    lang_map = fetch_language_mapping()
    with open(LANGUAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(lang_map, f, ensure_ascii=False, indent=2)
else:
    with open(LANGUAGE_FILE, 'r', encoding='utf-8') as f:
        lang_map = json.load(f)

# === 2. Fetch genre mapping ===
def get_genre_mapping():
    url = f"{BASE_URL}/genre/movie/list"
    params = {'api_key': API_KEY, 'language': 'en-US'}
    response = requests.get(url, params=params)
    return {g['id']: g['name'] for g in response.json().get('genres', [])}

genre_map = get_genre_mapping()

# === 3. Download TMDb movie data ===
movie_dict = {}
years = range(1980, 2026)

for year in years:
    for page in range(1, 101):
        url = f"{BASE_URL}/discover/movie"
        params = {
            'api_key': API_KEY,
            'sort_by': 'popularity.desc',
            'primary_release_year': year,
            'page': page
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            break
        results = response.json().get('results', [])
        if not results:
            break

        for movie in results:
            if movie['id'] not in movie_dict:
                movie['genre_names'] = [genre_map.get(gid, 'Unknown') for gid in movie.get('genre_ids', [])]
                movie['original_language_full'] = lang_map.get(movie.get('original_language', ''), 'Unknown')
                movie_dict[movie['id']] = movie

        print(f"📅 {year} Page {page} — Total Movies: {len(movie_dict)}")
        time.sleep(0.25)  # Respect TMDb rate limit

# === 4. Save TMDb movie data ===
with open(MOVIE_FILE, 'w', encoding='utf-8') as f:
    json.dump(list(movie_dict.values()), f, ensure_ascii=False, indent=2)

# === 5. Load MovieCine ratings ===
if not os.path.exists(RATINGS_FILE):
    raise FileNotFoundError("MovieCine ratings file not found!")

ratings_df = pd.read_csv(RATINGS_FILE)
print(f"📊 Loaded {len(ratings_df)} ratings from MovieCine")

# === 6. Merge TMDb metadata with ratings ===
movies_df = pd.DataFrame(movie_dict.values())
movies_df = movies_df[['id', 'title', 'release_date', 'genre_names', 'original_language_full']]
movies_df.rename(columns={'id': 'movie_id'}, inplace=True)

merged_df = ratings_df.merge(movies_df, on='movie_id', how='inner')
merged_df.dropna(subset=['title', 'rating'], inplace=True)

merged_df.to_csv(MERGED_FILE, index=False)
print(f"✅ Merged dataset saved as: {MERGED_FILE} ({len(merged_df)} rows)")
