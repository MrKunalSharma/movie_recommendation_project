import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import difflib

# Page Configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix-inspired theme
st.markdown("""
    <style>
    .main {
        background-color: #141414;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .movie-card {
        background-color: #282828;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .movie-info {
        color: #cccccc;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Load API Keys
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except Exception as e:
    st.error("Error loading API keys. Please check your secrets.toml file.")
    TMDB_API_KEY = None
    YOUTUBE_API_KEY = None

# Cache data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/movies.csv')
        # Fill NaN values
        df = df.fillna('')
        
        # Convert release_date to datetime and extract year
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Cache movie poster fetching
@st.cache_data
def fetch_poster(movie_id):
    if not TMDB_API_KEY or not movie_id:
        return None
    try:
        # First try to get movie details
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"
        
        # If no poster, try searching by title
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        search_response = requests.get(search_url)
        search_data = search_response.json()
        
        if 'results' in search_data and search_data['results']:
            poster_path = search_data['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception as e:
        st.warning(f"Error fetching poster: {e}")
    return None

@st.cache_data
def get_tmdb_id(movie_title, release_year=None):
    if not TMDB_API_KEY:
        return None
    try:
        query = movie_title
        if release_year:
            query += f" {release_year}"
            
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
        response = requests.get(search_url)
        data = response.json()
        
        if 'results' in data and data['results']:
            return data['results'][0]['id']
    except Exception as e:
        st.warning(f"Error finding TMDB ID: {e}")
    return None

# Get movie recommendations
def get_recommendations(movie_title, df, n_recommendations=10):
    # Fill NaN values with empty strings
    df = df.fillna('')
    
    # Prepare TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Combine features for similarity calculation
    df['combined_features'] = (df['genres'].astype(str) + ' ' + 
                             df['keywords'].astype(str) + ' ' + 
                             df['overview'].astype(str))
    
    # Create TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    try:
        # Get index of movie
        idx = df[df['title'] == movie_title].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
        
        return df.iloc[movie_indices]
    except IndexError:
        st.error(f"Movie '{movie_title}' not found in database.")
        return pd.DataFrame()

def main():
    # Debug: Test API keys
    st.write("Testing API Keys...")
    try:
        tmdb_key = st.secrets["TMDB_API_KEY"]
        youtube_key = st.secrets["YOUTUBE_API_KEY"]
        st.success("API keys loaded successfully!")
    except Exception as e:
        st.error(f"Error loading API keys: {e}")
        st.warning("Some features may not work without API keys")

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.title("🎬 Navigation")
        page = st.radio("Go to", ["Home", "Watchlist", "Statistics"])

    if page == "Home":
        # Main content
        st.title("🎬 Movie Recommender")
        
        # Search options
        search_method = st.selectbox(
            "How would you like to search?",
            ["Movie Name", "Filters"]
        )

        if search_method == "Movie Name":
            # Movie search with autocomplete
            movie_list = df['title'].tolist()
            movie_name = st.selectbox("Enter movie name:", movie_list)
            
            if movie_name:
                with st.spinner('Finding similar movies...'):
                    recommendations = get_recommendations(movie_name, df)
                    
                    if not recommendations.empty:
                        # Replace this entire section with the new code
                        for _, movie in recommendations.iterrows():
                            with st.container():
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    # Try to get TMDB ID if not present
                                    movie_id = movie.get('id') or get_tmdb_id(
                                        movie['title'], 
                                        str(movie['release_year']) if 'release_year' in movie else None
                                    )
                                    
                                    poster_url = fetch_poster(movie_id)
                                    if poster_url:
                                        st.image(poster_url, width=200)
                                    else:
                                        # Display default poster
                                        default_poster = f"""
                                        <div style="
                                            width: 200px;
                                            height: 300px;
                                            background-color: #333;
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            text-align: center;
                                            color: white;
                                            border-radius: 10px;
                                            padding: 10px;
                                        ">
                                            <h3>{movie['title']}</h3>
                                        </div>
                                        """
                                        st.markdown(default_poster, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"### {movie['title']}")
                                    st.write(f"**Genre:** {movie['genres']}")
                                    st.write(f"**Rating:** {movie['vote_average']}/10")
                                    st.write(f"**Release Year:** {movie['release_year']}")
                                    st.write(movie['overview'])
                                    
                                    # Add to watchlist button
                                    if st.button(f"Add to Watchlist", key=f"watch_{movie['title']}"):
                                        if movie['title'] not in st.session_state.watchlist:
                                            st.session_state.watchlist.append(movie['title'])
                                            st.success("Added to watchlist!")
                                        else:
                                            st.info("Already in watchlist!")

        elif search_method == "Filters":
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Get unique genres
                            all_genres = []
                            for genres in df['genres'].dropna():
                                all_genres.extend(genres.split('|'))
                            unique_genres = sorted(list(set(all_genres)))
                            
                            genres = st.multiselect(
                                "Select Genres",
                                unique_genres
                            )
                        
                        with col2:
                            min_rating = st.slider("Minimum Rating", 0.0, 10.0, 5.0)
                        
                        with col3:
                            # Modified year range slider
                            min_year = int(df['release_year'].min())
                            max_year = int(df['release_year'].max())
                            year_range = st.slider(
                                "Release Year",
                                min_value=min_year,
                                max_value=max_year,
                                value=(2000, 2020)
                            )
                        
                        # Filter movies
                        filtered_df = df.copy()
                        if genres:
                            filtered_df = filtered_df[filtered_df['genres'].str.contains('|'.join(genres), na=False)]
                        filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]
                        filtered_df = filtered_df[
                            filtered_df['release_year'].between(year_range[0], year_range[1])
                        ]
                        
                        # Display filtered movies
                        st.write(f"Found {len(filtered_df)} movies matching your criteria")
                        for _, movie in filtered_df.head(10).iterrows():
                            with st.container():
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    try:
                                        movie_id = movie.get('id') or get_tmdb_id(
                                            movie['title'], 
                                            str(movie['release_year']) if 'release_year' in movie else None
                                        )
                                        
                                        poster_url = fetch_poster(movie_id)
                                        if poster_url:
                                            st.image(poster_url, width=200)
                                        else:
                                            # Display default poster
                                            default_poster = f"""
                                            <div style="
                                                width: 200px;
                                                height: 300px;
                                                background-color: #333;
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                                text-align: center;
                                                color: white;
                                                border-radius: 10px;
                                                padding: 10px;
                                            ">
                                                <h3>{movie['title']}</h3>
                                            </div>
                                            """
                                            st.markdown(default_poster, unsafe_allow_html=True)
                                    except Exception as e:
                                        st.warning(f"Error displaying poster for {movie['title']}: {e}")
                                
                                with col2:
                                    st.markdown(f"### {movie['title']}")
                                    st.write(f"**Genre:** {movie['genres']}")
                                    st.write(f"**Rating:** {movie['vote_average']}/10")
                                    st.write(f"**Release Year:** {movie['release_year']}")
                                    st.write(movie['overview'])
                                    
                                    if st.button(f"Add to Watchlist", key=f"watch_filtered_{movie['title']}"):
                                        if movie['title'] not in st.session_state.watchlist:
                                            st.session_state.watchlist.append(movie['title'])
                                            st.success("Added to watchlist!")
                                        else:
                                            st.info("Already in watchlist!")

    elif page == "Watchlist":
                st.title("My Watchlist")
                if st.session_state.watchlist:
                    for movie in st.session_state.watchlist:
                        with st.container():
                            st.markdown(f"### {movie}")
                            if st.button("Remove", key=f"remove_{movie}"):
                                st.session_state.watchlist.remove(movie)
                                st.rerun()  # Changed from experimental_rerun to rerun
                else:
                    st.write("Your watchlist is empty!")

    elif page == "Statistics":
        st.title("Movie Statistics")
        
        # Rating distribution
        fig1 = px.histogram(
            df,
            x='vote_average',
            title='Rating Distribution',
            labels={'vote_average': 'Rating', 'count': 'Number of Movies'},
            nbins=20
        )
        st.plotly_chart(fig1)
        
        # Movies per year
        yearly_movies = df['release_year'].value_counts().sort_index()
        fig2 = px.line(
            x=yearly_movies.index,
            y=yearly_movies.values,
            title='Movies Released per Year',
            labels={'x': 'Year', 'y': 'Number of Movies'}
        )
        st.plotly_chart(fig2)
        
        # Genre distribution
        genre_counts = []
        for genres in df['genres'].dropna():
            genre_counts.extend(genres.split('|'))
        genre_df = pd.DataFrame(genre_counts, columns=['genre']).value_counts().reset_index()
        genre_df.columns = ['Genre', 'Count']
        
        fig3 = px.bar(
            genre_df.head(10),
            x='Genre',
            y='Count',
            title='Top 10 Movie Genres'
        )
        st.plotly_chart(fig3)

    # Footer
    st.markdown("""
        <div style='text-align: center; color: grey; padding: 20px;'>
            Made with ❤️ by Your Name
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
