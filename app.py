import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor

# --- UI Setup ---
st.set_page_config(page_title="Hybrid Recommendation Engine", page_icon="🍿", layout="wide")

# Header Section
st.title("🍿 Production-Grade Recommendation Engine")
st.markdown("A two-stage Machine Learning architecture: **Candidate Generation (Item-CF)** $\\rightarrow$ **Ranking (Gradient Boosting)**.")

# --- Load Data ---
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.dat", sep="::", engine="python", names=["movie_id", "title", "genres"], encoding='latin1')
    ratings = pd.read_csv("ratings.dat", sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"], encoding='latin1')
    ratings = ratings.sample(n=100000, random_state=42) # Downsample to prevent server crash
    data = ratings.merge(movies, on="movie_id")
    return movies, ratings, data

with st.spinner("Initializing Database..."):
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

with st.spinner("Training Gradient Boosting Ranker..."):
    popular_movies, item_sim_df, ranker, user_avg, movie_avg, movie_pop = prepare_models(data, movies)

# --- Main App Layout ---
st.markdown("---")
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("👤 User Profile")
    valid_users = data['user_id'].unique().tolist()
    selected_user = st.selectbox("Select a User ID to analyze:", valid_users[:50])
    
    # Calculate User Stats
    user_history = data[data.user_id == selected_user]
    watched_count = len(user_history)
    u_avg = user_avg.get(selected_user, 0)
    
    # Display Metrics
    col1, col2 = st.columns(2)
    col1.metric(label="Movies Watched", value=watched_count)
    col2.metric(label="Average Rating Given", value=f"{u_avg:.2f} ⭐")
    
    st.caption("Recent favorites:")
    st.dataframe(user_history[['title', 'rating']].sort_values(by='rating', ascending=False).head(3), hide_index=True)

with right_col:
    st.subheader("🎯 Top 5 Targeted Recommendations")
    
    if st.button("Generate Ranked Predictions", type="primary", use_container_width=True):
        with st.spinner("Running candidate generation and ranking..."):
            watched = user_history["movie_id"].tolist()
            candidates = set(popular_movies.head(20)["movie_id"].tolist())
            
            for m in watched[:5]:
                if m in item_sim_df.index:
                    candidates.update(item_sim_df[m].sort_values(ascending=False).iloc[1:11].index.tolist())
                    
            scored = []
            for m_id in candidates:
                if m_id in watched: continue
                score = ranker.predict([[u_avg, movie_avg.get(m_id, 3.0), movie_pop.get(m_id, 0)]])[0]
                scored.append((m_id, score))
                
            scored = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
            
            # Format output table
            recs_df = pd.DataFrame(scored, columns=["movie_id", "Predicted_Rating"])
            recs_df = recs_df.merge(movies, on="movie_id")[["title", "genres", "Predicted_Rating"]]
            recs_df["Predicted_Rating"] = recs_df["Predicted_Rating"].round(2)
            recs_df.rename(columns={"title": "Movie Title", "genres": "Genre", "Predicted_Rating": "AI Predicted Rating ⭐"}, inplace=True)
            
            st.dataframe(recs_df, use_container_width=True, hide_index=True)

# Explainability Section
with st.expander("🛠️ How does this engine work? (Architecture Breakdown)"):
    st.write("""
    1. **Candidate Generation (High Recall):** The system first analyzes the user's historical preferences and utilizes an **Item-to-Item Collaborative Filtering** matrix (measuring Cosine Similarity) to identify ~50 highly relevant candidate movies.
    2. **Machine Learning Ranking (High Precision):** A **Gradient Boosting Regressor** takes over. It extracts dynamic features (user rating strictness, global movie reception, and total popularity) to predict the exact fractional rating the user will assign to the candidate. 
    3. **Final Output:** The list is sorted dynamically based on the model's highest confidence predictions.
    """)
