import streamlit as st
import pandas as pd
from pathlib import Path

# Setup Path
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TOP_SIMILAR_MOVIES_FILE = OUTPUTS_DIR / "top_similar_movies.csv"

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #ff4b4b;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .stSelectbox label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fafafa;
    }
    .movie-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
        border-left: 4px solid #ff4b4b;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    .movie-title {
        font-size: 1.25rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .movie-similarity {
        font-size: 1rem;
        color: #b0b0b0;
    }
    .rank-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if TOP_SIMILAR_MOVIES_FILE.exists():
        df = pd.read_csv(TOP_SIMILAR_MOVIES_FILE)
        return df
    return None

def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("Discover your next favorite movie using AI-powered semantic search.")
    st.markdown("---")
    
    df = load_data()
    
    if df is None:
        st.error(f"Could not find `{TOP_SIMILAR_MOVIES_FILE.name}`. Please run the `generate_bert_analysis.py` script first.")
        return

    movies_list = sorted(df['movie_title'].unique())
    
    st.sidebar.title("Controls")
    st.sidebar.markdown("---")
    
    selected_movie = st.sidebar.selectbox(
        "🔍 Search for a movie you like:",
        options=movies_list,
        index=movies_list.index("The Dark Knight") if "The Dark Knight" in movies_list else 0
    )
    
    num_recommendations = st.sidebar.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("This engine uses **Sentence-BERT** embeddings to find semantically similar movies based on their overviews.")

    if selected_movie:
        st.subheader(f"Because you liked **{selected_movie}**...")
        
        recommendations = df[df['movie_title'] == selected_movie].sort_values(by='rank').head(num_recommendations)
        
        if recommendations.empty:
            st.warning("No recommendations found for this movie.")
        else:
            col1, col2 = st.columns(2)
            for index, row in recommendations.iterrows():
                col = col1 if index % 2 == 0 else col2
                with col:
                    similarity_pct = f"{row['similarity_score'] * 100:.1f}%"
                    
                    card_html = f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            <span class="rank-badge">#{int(row['rank'])}</span>
                            {row['similar_movie_title']}
                        </div>
                        <div class="movie-similarity">
                            Similarity Match: <strong>{similarity_pct}</strong>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
