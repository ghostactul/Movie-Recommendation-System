import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import scipy.sparse as sp
from pathlib import Path

# Sentence Transformer + FAISS for RAG-style retrieval
from sentence_transformers import SentenceTransformer
import faiss

# Torch and custom model/data loading
import torch
from src.model import NCF
from src.data import load_merged_df, build_mappings
from src.content import get_topk_content

# Paths & constants
MOVIE_FILE = 'data/raw/tmdb_movies.json'  # adjust as needed
MERGED_FILE = 'data/processed/merged_movielens_tmdb.csv'
SAVED_DIR = 'saved_models'
TFIDF_VECT = os.path.join(SAVED_DIR, 'tfidf_vectorizer.joblib')
TFIDF_MATRIX = os.path.join(SAVED_DIR, 'tfidf_matrix.npz')
MAPPINGS = os.path.join(SAVED_DIR, 'mappings.joblib')
NCF_CHECKPOINT = os.path.join(SAVED_DIR, 'ncf_bpr_checkpoint.pt')  # from your training script
CONTENT_EMB_FILE = os.path.join(SAVED_DIR, 'content_embeddings.joblib')
EMB_MODEL_NAME = 'all-MiniLM-L6-v2'

st.set_page_config(page_title="CineSage Hybrid Movie Recommender", layout="wide")

@st.cache_data(show_spinner=True)
def load_data():
    df = load_merged_df(merged_path=MERGED_FILE, movies_json_path=MOVIE_FILE, ratings_path=MERGED_FILE)
    return df

@st.cache_resource(show_spinner=True)
def load_artifacts():
    # Load TF-IDF artifacts
    tfidf = joblib.load(TFIDF_VECT)
    tfidf_m = sp.load_npz(TFIDF_MATRIX)
    
    # Load mappings
    mappings = joblib.load(MAPPINGS)
    
    # Load trained NCF model
    ckpt = torch.load(NCF_CHECKPOINT, map_location='cpu')
    num_users = len(mappings['user2idx'])
    num_items = len(mappings['movie2idx'])
    model = NCF(num_users=num_users, num_items=num_items, embedding_dim=32)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load content embeddings
    content_embeddings = joblib.load(CONTENT_EMB_FILE)
    
    # Load sentence transformer for RAG retrieval
    embed_model = SentenceTransformer(EMB_MODEL_NAME)
    
    return {
        'tfidf': tfidf,
        'tfidf_m': tfidf_m,
        'mappings': mappings,
        'ncf': model,
        'content_embeddings': content_embeddings,
        'embed_model': embed_model
    }

@st.cache_resource(show_spinner=True)
def build_faiss_index(df, _embed_model, field='overview'):
    texts = df[field].fillna('').astype(str).tolist()
    embeddings = _embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings

def recommend_existing_user(user_id, model, mappings, df, top_k=10):
    user2idx = mappings['user2idx']
    movie2idx = mappings['movie2idx']
    idx2movie = {v:k for k,v in movie2idx.items()}
    
    if user_id not in user2idx:
        st.warning("User not found in dataset.")
        return pd.DataFrame()
    
    uid = user2idx[user_id]
    num_items = len(movie2idx)
    device = torch.device('cpu')
    
    # Create tensors for all candidate items
    users = torch.LongTensor([uid]*num_items)
    items = torch.LongTensor(list(range(num_items)))
    
    with torch.no_grad():
        preds = model(users, items).numpy()
    
    top_indices = preds.argsort()[::-1][:top_k]
    rec_movie_ids = [idx2movie[i] for i in top_indices]
    
    rec_df = df[df['movie_id'].isin(rec_movie_ids)][['movie_id', 'title', 'release_date']].drop_duplicates().head(top_k)
    return rec_df

def recommend_new_user(last_movie_id, df_unique, content_embeddings, top_k=10):
    from src.content import get_topk_content
    
    if last_movie_id not in df_unique['movie_id'].values:
        st.warning("Last movie id not found in dataset.")
        return pd.DataFrame()
    
    idx = df_unique.index[df_unique['movie_id'] == last_movie_id].tolist()[0]
    top_indices = get_topk_content(idx, content_embeddings, top_k + 1)
    
    # Exclude the input movie itself
    seen_ids = {last_movie_id}
    recs = []
    for i in top_indices:
        mid = int(df_unique.iloc[i]['movie_id'])
        if mid not in seen_ids:
            recs.append({
                'movie_id': mid,
                'title': df_unique.iloc[i]['title'],
                'release_date': df_unique.iloc[i].get('release_date', '')
            })
            seen_ids.add(mid)
        if len(recs) >= top_k:
            break
    
    return pd.DataFrame(recs)

# UI Setup
st.title("CineSage Hybrid Movie Recommender")

df = load_data()
artifacts = load_artifacts()

# Create unique movies dataframe for content recommendations
df_unique = df.drop_duplicates(subset=['movie_id']).reset_index(drop=True)

# Build FAISS index for RAG retrieval
faiss_index, faiss_embeddings = build_faiss_index(df_unique, artifacts['embed_model'], field='overview')

mappings = artifacts['mappings']
model = artifacts['ncf']
content_embeddings = artifacts['content_embeddings']

col1, col2 = st.columns([1, 2])

with col1:
    mode = st.radio("Select user type", ("Existing user", "New user / Cold-start"))
    
    if mode == "Existing user":
        user_input = st.text_input("Enter user ID (integer)", value="1")
        user_id = int(user_input) if user_input.isdigit() else None
    else:
        last_movie_title = st.text_input("Enter most recent movie title or movie ID")
        last_movie_id = None
    
    top_k = st.number_input("Number of recommendations (Top K)", min_value=1, max_value=50, value=10)
    
    st.markdown("---")
    
    st.markdown("### RAG / Document Retrieval")
    use_rag = st.checkbox("Enable RAG-style retrieval (plot summaries & cast info)", value=True)
    if use_rag:
        query_rag = st.text_input("RAG Query (e.g. 'Plot summary and main cast for The Matrix')", value="")
    else:
        query_rag = ""

with col2:
    if st.button("Get Recommendations"):
        if mode == "Existing user":
            if user_id is None:
                st.warning("Please enter a valid integer user ID.")
            else:
                recs_df = recommend_existing_user(user_id, model, mappings, df_unique, top_k=top_k)
                if recs_df.empty:
                    st.info("No recommendations found.")
                else:
                    st.subheader(f"Top {top_k} recommendations for user {user_id}")
                    st.table(recs_df)
        else:
            # resolve last_movie_id from title or id
            if last_movie_title:
                if last_movie_title.isdigit():
                    last_movie_id = int(last_movie_title)
                else:
                    candidates = df_unique[df_unique['title'].str.contains(last_movie_title, case=False, na=False)]
                    if not candidates.empty:
                        last_movie_id = int(candidates.iloc[0]['movie_id'])
                    else:
                        last_movie_id = None
                
                if last_movie_id is None:
                    st.warning("Could not find the movie by that title or ID.")
                else:
                    recs_df = recommend_new_user(last_movie_id, df_unique, content_embeddings, top_k=top_k)
                    if recs_df.empty:
                        st.info("No content-based recommendations found.")
                    else:
                        st.subheader(f"Top {top_k} content-based recommendations similar to movie ID {last_movie_id}")
                        st.table(recs_df)
            else:
                st.warning("Please enter a recent movie title or ID.")
    
    st.markdown("---")
    
    if use_rag and query_rag:
        q_emb = artifacts['embed_model'].encode([query_rag], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = faiss_index.search(q_emb, 5)
        
        st.subheader("Top retrieved documents (via FAISS):")
        retrieved = df_unique.iloc[I[0]][['movie_id', 'title', 'overview']]
        
        for _, row in retrieved.iterrows():
            st.markdown(f"**{row['title']}** (Movie ID: {row['movie_id']})")
            overview = row['overview'] if isinstance(row['overview'], str) else ""
            st.write(overview[:800] + ("..." if len(overview) > 800 else ""))
            st.markdown("---")
        
        st.info("Note: This is a retrieval-only step. For synthesis/generation, integrate an LLM (OpenAI/HuggingFace).")

st.markdown("---")
st.markdown("### Notes")
st.markdown("""
- This hybrid recommender combines Neural Collaborative Filtering (NCF) with content-based filtering using sentence embeddings.
- For cold-start users, recommendations are based on content similarity using movie genres and overviews.
- RAG retrieval uses FAISS + Sentence Transformer embeddings to find relevant movie documents for user queries.
- To enable full RAG-style generative answers, integrate with an LLM like OpenAI GPT or HuggingFace Transformers.
""")
