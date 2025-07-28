import streamlit as st
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import NCF

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have saved the trained model, load it
model = NCF(num_users, num_movies).to(device)

# Load the trained model (if saved previously)
# model.load_state_dict(torch.load("path_to_saved_model.pth"))

# For simplicity, we will just use the existing model from the current session

# Function to recommend movies based on a movie
def recommend_movies(movie_id, top_k=5):
    model.eval()

    # Get the movie embedding from the encoder
    movie_index = movie_enc.transform([movie_id])[0]  # Transform movie_id to index
    movie_tensor = torch.tensor([movie_index], dtype=torch.long).to(device)
    
    # Get all user embeddings (or top K recommendations)
    all_movie_ids = torch.tensor(np.arange(num_movies), dtype=torch.long).to(device)
    movie_predictions = model(torch.zeros(len(all_movie_ids), dtype=torch.long).to(device), all_movie_ids)

    # Get top K movie predictions
    _, top_indices = torch.topk(movie_predictions, top_k)
    
    # Decode the movie indices back to movie ids
    recommended_movie_ids = movie_enc.inverse_transform(top_indices.cpu().numpy())
    return recommended_movie_ids

# Streamlit App
st.title("CineSage Movie Recommendation Chatbot")

st.write("### Welcome! Ask for movie recommendations.")
st.write("I will suggest similar movies based on your favorite movie.")

# Input: Favorite Movie
user_input = st.text_input("Enter your favorite movie:")

if user_input:
    # Recommend movies based on user input
    try:
        recommendations = recommend_movies(user_input)
        st.write("### Recommended Movies:")
        for idx, movie in enumerate(recommendations, start=1):
            st.write(f"{idx}. {movie}")
    except Exception as e:
        st.error(f"Sorry, an error occurred: {e}")