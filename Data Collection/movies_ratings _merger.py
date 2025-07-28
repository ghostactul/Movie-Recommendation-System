import pandas as pd
import json

# === File Paths ===
MOVIE_FILE = 'tmdb_movies.json'
RATINGS_FILE = './ml-32m/ratings.csv'
MERGED_FILE = 'merged_moviecine_tmdb.csv'

# === Step 1: Load TMDb Movies JSON ===
with open(MOVIE_FILE, 'r', encoding='utf-8') as f:
    tmdb_movies = json.load(f)

# Convert to DataFrame
movies_df = pd.DataFrame(tmdb_movies)

# Rename 'id' to 'movie_id' to match ratings
movies_df.rename(columns={'id': 'movie_id'}, inplace=True)

# Optional: Keep only relevant columns
movies_df = movies_df[['movie_id', 'title', 'release_date', 'genre_names', 'original_language_full']]

print(f"🎬 Loaded {len(movies_df)} TMDb movies.")

# === Step 2: Load MovieCine Ratings CSV ===
ratings_df = pd.read_csv(RATINGS_FILE)

# Rename column for merge compatibility
ratings_df.rename(columns={'movieId': 'movie_id', 'userId': 'user_id'}, inplace=True)

print(f"📊 Loaded {len(ratings_df)} user ratings.")

# === Step 3: Merge Ratings with Movie Metadata ===
merged_df = ratings_df.merge(movies_df, on='movie_id', how='inner')

# Drop rows without titles or ratings (just in case)
merged_df.dropna(subset=['title', 'rating'], inplace=True)

# === Step 4: Save Merged Data ===
merged_df.to_csv(MERGED_FILE, index=False)

print(f"✅ Final merged dataset saved: {MERGED_FILE}")
print(f"🧩 Rows: {len(merged_df)}, Users: {merged_df['user_id'].nunique()}, Movies: {merged_df['movie_id'].nunique()}")
