import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Hybrid Movie Ranker", layout="wide")
st.title("🎬 Production-Grade Two-Stage Recommender")

# --- Load Data ---
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.dat", sep="::", engine="python", names=["movie_id", "title", "genres"], encoding='latin1')
    ratings = pd.read_csv("ratings.dat", sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"], encoding='latin1')
    ratings = ratings.sample(n=100000, random_state=42) # Downsample to prevent server crash
    data = ratings.merge(movies, on="movie_id")
    return movies, ratings, data

with st.spinner("Loading MovieLens Data..."):
    movies, ratings, data = load_data()

# --- Precompute & Train ---
@st.cache_resource
def prepare_models(_data, _movies):
    movie_stats = _data.groupby("movie_id").agg(avg_rating=("rating", "mean"), rating_count=("rating", "count")).reset_index()
    movie_stats["popularity_score"] = movie_stats["avg_rating"] * movie_stats["rating_count"]
    popular_movies = movie_stats.sort_values("popularity_score", ascending=False).merge(_movies, on="movie_id")
    
    user_avg = _data.groupby("user_id")["rating"].mean().to_dict()
    movie_avg = movie_stats.set_index("movie_id")["avg_rating"].to_dict()
    movie_pop = movie_stats.set_index("movie_id")["rating_count"].to_dict()

    item_user_matrix = _data.pivot_table(index="movie_id", columns="user_id", values="rating").fillna(0)
    item_sim_df = pd.DataFrame(cosine_similarity(item_user_matrix), index=item_user_matrix.index, columns=item_user_matrix.index)

    X, y = [], []
    for _, row in _data.sample(10000, random_state=42).iterrows():
        X.append([user_avg.get(row.user_id, 3.0), movie_avg.get(row.movie_id, 3.0), movie_pop.get(row.movie_id, 0)])
        y.append(row.rating)
        
    ranker = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    ranker.fit(X, y)

    return popular_movies, item_sim_df, ranker, user_avg, movie_avg, movie_pop

with st.spinner("Compiling Neural Matrices & Training Ranker..."):
    popular_movies, item_sim_df, ranker, user_avg, movie_avg, movie_pop = prepare_models(data, movies)

# --- UI & Logic ---
valid_users = data['user_id'].unique().tolist()
selected_user = st.sidebar.selectbox("Select User ID:", valid_users[:50])

if st.button("Generate Recommendations", type="primary"):
    watched = data[data.user_id == selected_user]["movie_id"].tolist()
    candidates = set(popular_movies.head(20)["movie_id"].tolist())
    
    for m in watched[:5]:
        if m in item_sim_df.index:
            candidates.update(item_sim_df[m].sort_values(ascending=False).iloc[1:11].index.tolist())
            
    scored = []
    for m_id in candidates:
        if m_id in watched: continue
        score = ranker.predict([[user_avg.get(selected_user, 3.0), movie_avg.get(m_id, 3.0), movie_pop.get(m_id, 0)]])[0]
        scored.append((m_id, score))
        
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
    recs_df = pd.DataFrame(scored, columns=["movie_id", "Predicted Rating"]).merge(movies, on="movie_id")[["title", "genres", "Predicted Rating"]]
    
    st.dataframe(recs_df.style.background_gradient(subset=['Predicted Rating'], cmap='Blues'), use_container_width=True)
