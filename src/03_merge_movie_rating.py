import pandas as pd
import json
import os

# File Paths
MOVIE_FILE = 'data/raw/tmdb_movies.json'           
RATINGS_FILE = 'data/raw/ml-latest-small/ratings.csv'  
MERGED_FILE = 'data/processed/merged_movielens_tmdb.csv'

# Step 1: Load TMDb Movies JSON 
if not os.path.exists(MOVIE_FILE):
    raise FileNotFoundError(f"TMDb movie file not found: {MOVIE_FILE}")

with open(MOVIE_FILE, 'r', encoding='utf-8') as f:
    tmdb_movies = json.load(f)

movies_df = pd.DataFrame(tmdb_movies)

# Rename 'id' to 'movie_id' to match MovieLens ratings
movies_df.rename(columns={'id': 'movie_id'}, inplace=True)

# Keep only relevant columns
movies_df = movies_df[['movie_id', 'title', 'release_date', 'genre_names', 'original_language_full', 'overview']]

print(f"Loaded {len(movies_df)} TMDb movies.")

# Step 2: Load MovieLens Ratings CSV 
if not os.path.exists(RATINGS_FILE):
    raise FileNotFoundError(f"MovieLens ratings file not found: {RATINGS_FILE}")

ratings_df = pd.read_csv(RATINGS_FILE)

# MovieLens uses movieId, userId, rating, timestamp
ratings_df.rename(columns={'movieId': 'movie_id', 'userId': 'user_id'}, inplace=True)

print(f"Loaded {len(ratings_df)} MovieLens ratings.")

# Step 3: Merge Ratings with TMDb Metadata
merged_df = ratings_df.merge(movies_df, on='movie_id', how='inner')

# Drop rows without titles or ratings
merged_df.dropna(subset=['title', 'rating'], inplace=True)

# Step 4: Save Merged Data
merged_df.to_csv(MERGED_FILE, index=False)

print(f"Final merged dataset saved: {MERGED_FILE}")
print(f"Rows: {len(merged_df)}, Users: {merged_df['user_id'].nunique()}, Movies: {merged_df['movie_id'].nunique()}")
