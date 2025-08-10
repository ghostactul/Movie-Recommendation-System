import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def load_merged_df(merged_path='data/processed/merged_movielens_tmdb.csv',
                   ratings_path='data/raw/ml-latest-small/ratings.csv',
                   movies_json_path='data/raw/tmdb_movies.json'):
    """Load merged file if exists, otherwise attempt to join minimal data."""
    mp = Path(merged_path)
    if mp.exists():
        df = pd.read_csv(mp)
        return df
    # Fallback: try to build a minimal merged df from ratings + movies_json
    ratings = pd.read_csv(ratings_path)
    with open(movies_json_path, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    movies_df = pd.DataFrame(movies)
    # Expect movies_df to have 'id','title','release_date','genre_names','overview','original_language_full'
    movies_df = movies_df.rename(columns={'id':'movie_id'})
    merged = ratings.merge(movies_df, how='left', on='movie_id')
    return merged

def build_mappings(df):
    user_ids = sorted(df['user_id'].unique().tolist())
    movie_ids = sorted(df['movie_id'].unique().tolist())
    user2idx = {u:i for i,u in enumerate(user_ids)}
    idx2user = {i:u for u,i in user2idx.items()}
    movie2idx = {m:i for i,m in enumerate(movie_ids)}
    idx2movie = {i:m for m,i in movie2idx.items()}
    return user2idx, idx2user, movie2idx, idx2movie
