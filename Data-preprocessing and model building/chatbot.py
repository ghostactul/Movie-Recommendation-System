import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import Accelerator
from model import NCF


st.set_page_config(page_title="🎬 CineSage", layout="centered")
# Custom background and styling
st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        background: linear-gradient(to bottom right, #1e3c72, #2a5298) !important;
        color: white !important;
        box-shadow: none !important;
    }

    header[data-testid="stHeader"] * {
        color: white !important;
    }

    /* Optional: Make toolbar icons more visible */
    [data-testid="stToolbar"] {
        background-color: transparent !important;
        color: white !important;
    }
    /* Global gradient background */
    html, body, .stApp {
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        background: linear-gradient(to bottom right, #1e3c72, #2a5298) !important;
        background-attachment: fixed !important;
        background-size: cover !important;
        color: #f1f1f1;
        font-family: 'Segoe UI', sans-serif;
        overflow-x: hidden;
    }

    /* Layout */
    .block-container {
        background-color: transparent !important;
        padding-top: 3rem !important;
    }

    section.main {
        background-color: transparent !important;
        min-height: 100vh !important;
    }

    /* Chat input bar */
    .stChatInput {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        width: 100% !important;
        z-index: 1000;
        background: #1b1f3b !important; /* Deep contrasting purple/navy */
        border-top: 1px solid #444 !important;
        display: flex !important;
        justify-content: center !important;
        padding: 18px 0 !important;
        box-shadow: 0 -2px 12px rgba(0, 0, 0, 0.4);
    }

    .stChatInput > div {
        width: 60% !important;
        max-width: 720px;
    }

    /* Chat input field */
    section[tabindex="0"] textarea {
        background: linear-gradient(to right, #2e335a, #1c1c2e) !important;
        color: #ffffff !important;
        border: 1px solid #5b5e8d !important;
        border-radius: 25px !important;
        padding: 12px 20px !important;
        box-shadow: 0 0 6px rgba(91, 94, 141, 0.4);
        font-size: 16px !important;
    }

    section[tabindex="0"] textarea::placeholder {
        color: #bbbbbb !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background-color: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)




# --- Load LLaMA 3.1 8B model via Hugging Face with your access token ---
@st.cache_resource
def load_llm():
    model_name = "meta-llama/Llama-3.2-1B"
    access_token = ""  

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=access_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise RuntimeError("Failed to load the LLaMA model.")

llm_pipeline = load_llm()

# --- Load Resources ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("movie_encoder.pkl", "rb") as f:
    movie_enc = pickle.load(f)
with open("user_encoder.pkl", "rb") as f:
    user_enc = pickle.load(f)

movie_df = pd.read_csv("../Datasets/merged_moviecine_tmdb.csv")  # Includes 'movie_id', 'title'
with open("../Datasets/tmdb_movies.json", "r", encoding="utf-8") as f:
    movie_meta = {entry["title"]: entry for entry in json.load(f)}

num_users = len(user_enc.classes_)
num_movies = len(movie_enc.classes_)

model = NCF(num_users, num_movies).to(device)
model.load_state_dict(torch.load("ncf_model.pth", map_location=device))
model.eval()

# --- LLM Helpers ---
def ask_llm(prompt, max_tokens=150):
    try:
        response = llm_pipeline(prompt, max_new_tokens=max_tokens)[0]["generated_text"]

        # Robust prompt trimming: split where the prompt ends or summary begins
        if "Summary:" in response:
            response = response.split("Summary:")[-1].strip()

        return response.strip()
    except Exception as e:
        raise RuntimeError(f"Error processing the LLaMA response: {e}")

import re

def extract_movie_title(user_input, all_titles):
    """
    Extract the best-matching movie title from the user input using whole-word boundary search.
    """
    user_input_lower = user_input.lower()
    for title in sorted(all_titles, key=len, reverse=True):  # prioritize longer matches
        pattern = r'\b' + re.escape(title.lower()) + r'\b'
        if re.search(pattern, user_input_lower):
            return title
    return None


def summarize_plot(title):
    plot = movie_meta.get(title, {}).get("overview", "")
    if not plot or len(plot.strip()) < 20:  # minimum length check
        return f"No plot found for '{title}'. Please check the title or try another movie."

    prompt = f"Summarize the following movie plot in exactly 3 complete and engaging sentences:\n{plot}\n\nSummary:"

    response = ask_llm(prompt, max_tokens=300)

    # Keep only 2–3 sentences
    sentences = response.split('. ')
    summary = '. '.join(sentences[:3]).strip()
    if not summary.endswith('.'):
        summary += '.'

    return summary

def cast_crew_details(title):
    cast_data = movie_meta.get(title, {}).get("cast", "")
    if not cast_data:
        return "No cast and crew data available."
    prompt = f"List main cast and crew of the movie '{title}': {cast_data}\nAnswer:"
    return ask_llm(prompt, 100)

def explain_recommendation(base_title, recommended_title):
    base_plot = movie_meta.get(base_title, {}).get("overview", "")
    rec_plot = movie_meta.get(recommended_title, {}).get("overview", "")
    if not base_plot or not rec_plot:
        return "Explanation not available due to missing plot data."
    prompt = (
        f"Explain why someone who liked '{base_title}' might also like '{recommended_title}'.\n"
        f"Plot of {base_title}: {base_plot}\n"
        f"Plot of {recommended_title}: {rec_plot}\n\nExplanation:"
    )
    return ask_llm(prompt, 150)

# --- Recommendation Logic ---
def get_encoded_movie_index(movie_title):
    try:
        movie_id = movie_df[movie_df['title'].str.lower() == movie_title.lower()]['movie_id'].values[0]
        return movie_enc.transform([movie_id])[0]
    except IndexError:
        raise ValueError("Movie not found in dataset.")

def recommend_movies(movie_title, top_k=5):
    try:
        movie_index = get_encoded_movie_index(movie_title)

        liked_embedding = model.item_embed_gmf(torch.tensor(movie_index).to(device))

        all_embeddings = model.item_embed_gmf.weight.data

        similarities = torch.nn.functional.cosine_similarity(
            liked_embedding.unsqueeze(0), all_embeddings
        )

        similarities[movie_index] = -1  # Exclude self

        top_k_indices = torch.topk(similarities, k=top_k * 5).indices.cpu().numpy()  # Over-select to remove dups

        recommended_movie_ids = movie_enc.inverse_transform(top_k_indices)
        unique_titles = []
        for movie_id in recommended_movie_ids:
            title = movie_df.loc[movie_df['movie_id'] == movie_id, 'title'].values[0]
            if title not in unique_titles:
                unique_titles.append(title)
            if len(unique_titles) == top_k:
                break

        return unique_titles

    except Exception as e:
        raise RuntimeError(f"Error in recommending movies: {e}")


# --- Streamlit UI ---

st.markdown("""
    <h1 style='text-align: center; color: #a3c9f1;'>🎬 CineSage: AI Movie Recommender</h1>
    <p style='text-align: center; color: #dbe9f7;'>Discover movies you'll love. Ask anything about cast, summary, or recommendations!</p>
    <hr style='border-top: 1px solid #bbb; margin-top: 10px; margin-bottom: 25px;'>
""", unsafe_allow_html=True)

# Store session state for conversation context
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# --- Handle User Input via st.chat_input ---
user_input = st.chat_input("🎥 Ask something or tell me a movie you like:")

if user_input:
    # Store user input in conversation history
    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    # Check if the user has inputted a movie title
    recommendations = []
    try:
        # Check if the input is a movie title (simple matching)
        movie_titles = movie_df['title'].str.lower().tolist()
        if any(user_input.lower() in title for title in movie_titles):
            recommendations = recommend_movies(user_input)
            st.session_state.recommendations = recommendations
            st.session_state.base_movie = user_input
            recommendation_text = "\n".join([f"{i}. 🎬 <b>{title}</b>" for i, title in enumerate(recommendations, 1)])
            styled_block = f"""
            <div style='background-color: rgba(72, 133, 237, 0.2); color: #ffffff; padding: 15px; border-left: 5px solid #4285F4; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);'>
            <b> Here are top 5 movie recommendations based on '{user_input}':</b><br><br>{recommendation_text}
            </div>
            """
            st.session_state.conversation_history.append({"role": "bot", "content": styled_block})
        
        # Add a follow-up prompt for more details like summary, cast, or why recommended
        if "summary" in user_input.lower() or "cast" in user_input.lower() or "why" in user_input.lower():
            lower = user_input.lower()
            movie_title = None

            # Check if the user mentioned a movie title
            movie_title = extract_movie_title(user_input, movie_df['title'].tolist())

            if movie_title:
                if "summary" in lower:
                    summary = summarize_plot(movie_title)
                    summary_card = f"""
                    <div style='background-color: rgba(0, 230, 118, 0.2); color: #e0ffe0; padding: 12px 18px; border-left: 5px solid #00e676; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);'>
                    <b> Summary of '{movie_title}':</b><br>{summary}
                    </div>
                    """
                    st.session_state.conversation_history.append({"role": "bot", "content": summary_card})

                elif "cast" in lower or "crew" in lower:
                    cast_details = cast_crew_details(movie_title)
                    cast_card = f"""
                    <div style='background-color: rgba(255, 87, 34, 0.2); color: #ffe5d0; padding: 12px 18px; border-left: 5px solid #ff5722; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);'>
                    <b>👥 Cast and Crew of '{movie_title}':</b><br>{cast_details}
                    </div>
                    """
                    st.session_state.conversation_history.append({"role": "bot", "content": cast_card})
                elif "why" in lower or "recommend" in lower:
                    base_movie = st.session_state.base_movie
                    explanation = explain_recommendation(base_movie, movie_title)
                    explanation_card = f"""
                    <div style='background-color: rgba(156, 39, 176, 0.2); color: #f3e5f5; padding: 12px 18px; border-left: 5px solid #9c27b0; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.25);'>
                    <b>Why was '{movie_title}' recommended?</b><br>{explanation}
                    </div>
                    """
                    st.session_state.conversation_history.append({"role": "bot", "content": explanation_card})
    except Exception as e:
        st.exception(e)

# --- Display Chat History ---
for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user").markdown(f" {message['content']}")
    else:
        # Use HTML-rendering for bot responses
        st.markdown(message["content"], unsafe_allow_html=True)
        
st.markdown("""
<hr>
<div style='text-align: center; font-size: 0.85em; color: #999; margin-top: 20px;'>
Made with ❤️ by CineSage • Powered by LLaMA + Streamlit
</div>
""", unsafe_allow_html=True)
