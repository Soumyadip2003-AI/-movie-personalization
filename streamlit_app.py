import numpy as np

    np._import_array()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import streamlit as st

# Load Datasets
@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv("rating.csv")
        movies = pd.read_csv("movie.csv")
        data = pd.merge(ratings, movies, on="movieId")
        return ratings, movies, data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

ratings, movies, data = load_data()

if ratings is None or movies is None or data is None:
    st.stop()

# Collaborative Filtering using Surprise SVD
@st.cache_resource
def train_collaborative_filtering_model():
    try:
        reader = Reader(rating_scale=(0.5, 5.0))
        surprise_data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        trainset, testset = train_test_split(surprise_data, test_size=0.2)
        model = SVD()
        model.fit(trainset)
        return model
    except Exception as e:
        st.error(f"Error training collaborative filtering model: {e}")
        return None

model = train_collaborative_filtering_model()

if model is None:
    st.stop()

def recommend_collaborative(user_id, num_recommendations=10):
    user_ratings = data[data['userId'] == user_id]
    watched_movies = user_ratings['movieId'].tolist()
    all_movie_ids = movies['movieId'].tolist()

    # Predict ratings for movies the user hasn't watched
    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in all_movie_ids if movie_id not in watched_movies
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [pred[0] for pred in predictions[:num_recommendations]]
    return movies[movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()

# Content-Based Filtering
@st.cache_resource
def build_content_based_model():
    try:
        tfidf = TfidfVectorizer(stop_words="english")
        movies['genres'] = movies['genres'].fillna("")
        tfidf_matrix = tfidf.fit_transform(movies['genres'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        return cosine_sim
    except Exception as e:
        st.error(f"Error building content-based model: {e}")
        return None

cosine_sim = build_content_based_model()

if cosine_sim is None:
    st.stop()

def recommend_content_based(movie_title, num_recommendations=10):
    try:
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices]
    except Exception as e:
        st.error(f"Error generating content-based recommendations: {e}")
        return []

# Streamlit App
st.title("Personalized Movie Recommendation System")

# Select Recommendation Type
st.sidebar.title("Choose Recommendation Type")
rec_type = st.sidebar.radio("Recommendation Type", ["Content-Based", "Collaborative Filtering"])

if rec_type == "Content-Based":
    st.subheader("Content-Based Recommendations")
    movie_title = st.selectbox("Choose a movie:", movies['title'].values)
    if st.button("Recommend Movies"):
        recommendations = recommend_content_based(movie_title)
        if recommendations:
            st.write("Recommended Movies:")
            for movie in recommendations:
                st.write(movie)
        else:
            st.write("No recommendations found.")

elif rec_type == "Collaborative Filtering":
    st.subheader("Collaborative Filtering Recommendations")
    user_id = st.number_input("Enter User ID:", min_value=int(ratings['userId'].min()), max_value=int(ratings['userId'].max()))
    if st.button("Recommend Movies"):
        recommendations = recommend_collaborative(int(user_id))
        if recommendations:
            st.write("Recommended Movies:")
            for movie in recommendations:
                st.write(movie)
        else:
            st.write("No recommendations found.")
