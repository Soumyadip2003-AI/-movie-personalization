import numpy as np
try:
    np._import_array()
except AttributeError:
    pass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import streamlit as st

@st.cache_data
def load_data():
    ratings = pd.read_csv("rating.csv")
    movies = pd.read_csv("movie.csv")
    data = pd.merge(ratings, movies, on="movieId")
    return ratings, movies, data

ratings, movies, data = load_data()

@st.cache_data
def train_collaborative_filtering_model():
    reader = Reader(rating_scale=(0.5, 5.0))
    surprise_data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(surprise_data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model

model = train_collaborative_filtering_model()

def recommend_collaborative(user_id, num_recommendations=10):
    if user_id not in data['userId'].unique():
        return ["No recommendations available for this user."]
    user_ratings = data[data['userId'] == user_id]
    watched_movies = user_ratings['movieId'].tolist()
    all_movie_ids = movies['movieId'].tolist()
    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in all_movie_ids if movie_id not in watched_movies
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [pred[0] for pred in predictions[:num_recommendations]]
    return movies[movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()

@st.cache_data
def build_content_based_model():
    tfidf = TfidfVectorizer(stop_words="english")
    movies['genres'] = movies['genres'].fillna("")
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

cosine_sim = build_content_based_model()

def recommend_content_based(movie_title, num_recommendations=10):
    if movie_title not in movies['title'].values:
        return ["Movie not found in dataset."]
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

st.title("Personalized Movie Recommendation System")
st.sidebar.title("Choose Recommendation Type")
rec_type = st.sidebar.radio("Recommendation Type", ["Content-Based", "Collaborative Filtering"])

if rec_type == "Content-Based":
    st.subheader("Content-Based Recommendations")
    movie_title = st.selectbox("Choose a movie:", movies['title'].values)
    if st.button("Recommend Movies"):
        recommendations = recommend_content_based(movie_title)
        st.write("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)

elif rec_type == "Collaborative Filtering":
    st.subheader("Collaborative Filtering Recommendations")
    user_id = st.number_input("Enter User ID:", min_value=1, max_value=ratings['userId'].max())
    if st.button("Recommend Movies"):
        recommendations = recommend_collaborative(int(user_id))
        st.write("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)
