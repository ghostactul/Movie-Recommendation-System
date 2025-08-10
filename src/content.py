import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

def build_sentence_embeddings(df, field='content', model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    sentences = df[field].fillna('').tolist()
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def get_topk_content(idx, embeddings, k=10):
    query_vec = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return top_idx.tolist()

