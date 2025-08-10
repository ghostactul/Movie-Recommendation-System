import joblib
from pathlib import Path
import pandas as pd

# Your load function to get the merged dataframe (adjust if needed)
from src.data import load_merged_df, build_mappings

MERGED_FILE = 'data/processed/merged_movielens_tmdb.csv'
RATINGS_FILE = 'data/raw/ml-latest-small/ratings.csv'  # adjust if needed
MOVIE_FILE = 'data/raw/ml-latest-small/ratings.csv'   # adjust if needed

# Load merged dataframe (adjust parameters if your function differs)
df = load_merged_df(merged_path=MERGED_FILE, ratings_path=RATINGS_FILE, movies_json_path=MOVIE_FILE)

# Build mappings
user2idx, idx2user, movie2idx, idx2movie = build_mappings(df)

# Save mappings dictionary as a single joblib file
mappings = {
    'user2idx': user2idx,
    'idx2user': idx2user,
    'movie2idx': movie2idx,
    'idx2movie': idx2movie
}

save_path = Path('saved_models')
save_path.mkdir(exist_ok=True)

joblib.dump(mappings, save_path / 'mappings.joblib')

print("Mappings saved to", save_path / 'mappings.joblib')
