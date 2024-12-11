import pandas as pd
import ast
import requests
from sentence_transformers import SentenceTransformer
import streamlit as st
import faiss
import numpy as np

# TMDB API Key
API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# Fetch poster URL from TMDB API
@st.cache_data
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        tmdb_url = f"https://www.themoviedb.org/movie/{movie_id}"
        return (f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None, tmdb_url)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None, None

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('D:/Devsinc/Data Science Work/Content Recommendation/tmdb_5000_movies.csv')
    credits = pd.read_csv('D:/Devsinc/Data Science Work/Content Recommendation/tmdb_5000_credits.csv')
    return movies, credits

# Merge and preprocess data
@st.cache_data
def preprocess_data(movies, credits):
    data = movies.merge(credits, on='title', how='left')  # Left join to ensure we have all movie data
    data = data[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date', 'runtime', 'vote_average']].dropna(subset=['overview'])

    # Convert JSON-like strings to lists
    def convert_to_list(col):
        try:
            return [i['name'] for i in ast.literal_eval(col)]
        except:
            return []

    data['genres'] = data['genres'].apply(convert_to_list)
    data['keywords'] = data['keywords'].apply(convert_to_list)
    data['actors'] = data['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:10])
    data['crew'] = data['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] in {'Director', 'Screenplay'}])

    for col in ['genres', 'keywords', 'actors', 'crew']:
        data[col] = data[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # Combine features into a single text column
    data['soup'] = data.apply(
        lambda row: " ".join(row['overview'].split() + row['genres'] + row['keywords'] + row['actors'] + row['crew']),
        axis=1
    )
    return data

# Load Sentence-BERT model (Cached)
@st.cache_resource
def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

# Compute embeddings (Cached)
@st.cache_resource
def compute_embeddings(data, _model):  # Renamed 'model' to '_model'
    embeddings = _model.encode(data['soup'].tolist(), show_progress_bar=True)
    return np.array(embeddings)

# Recommend movies
def recommend(movie_name, data, model, similarity_index, selected_genre=""):
    movie_name = movie_name.lower()
    matching_movies = data[data['title'].str.lower().str.contains(movie_name)]

    if matching_movies.empty:
        st.error(f"Movie titled '{movie_name}' not found.")
        possible_matches = data[data['title'].str.contains(movie_name, case=False, na=False)]['title'].head(5)
        st.info(f"Suggested titles: {', '.join(possible_matches)}")
        return [], []
    else:
        idx = matching_movies.index[0]

        # Ensure the input to FAISS is a 2D array
        query_embedding = model.encode([data['soup'].iloc[idx]])  # Get the movie embedding
        query_embedding = query_embedding.reshape(1, -1)  # Ensure 2D shape

        # Search for similar movies using FAISS index
        distances, indices = similarity_index.search(query_embedding, 21)  # Get top 20 similar movies
        movie_indices = indices[0][1:]  # Exclude the movie itself

        # Ensure indices are within the bounds of the data
        valid_indices = [i for i in movie_indices if i < len(data)]

        if not valid_indices:
            st.error("No valid movie recommendations found.")
            return [], []

        # Get recommended movie titles and posters using valid indices
        recommended_titles = data['title'].iloc[valid_indices].tolist()
        recommended_posters_links = [fetch_poster(data['id'].iloc[i]) for i in valid_indices]

        # If genre is selected, filter recommendations by genre
        if selected_genre != "All":
            data_filtered = data[data['genres'].apply(lambda x: selected_genre in x)]
            recommended_titles = [title for title in recommended_titles if any(selected_genre in x for x in data_filtered['genres'].iloc[valid_indices])]
            recommended_posters_links = [poster for i, poster in enumerate(recommended_posters_links) if any(selected_genre in x for x in data_filtered['genres'].iloc[valid_indices])]

        return recommended_titles, recommended_posters_links


# Main function
def main():
    st.title("🎬 Movie Recommendation System")

    # Load data and preprocess before accessing it
    movies, credits = load_data()
    data = preprocess_data(movies, credits)

    # Load Sentence-BERT model
    model = load_model()

    # Compute embeddings (without caching)
    embeddings = compute_embeddings(data, model)

    # Create FAISS index for fast similarity search
    similarity_index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    similarity_index.add(embeddings)  # Add movie embeddings to the index

    # Create tabs
    tab1, tab2 = st.tabs(["Recommendation", "All Movies"])

    # Genre selection outside of both tabs
    all_genres = sorted(set([genre for sublist in data['genres'] for genre in sublist]))  # Now 'data' is defined
    selected_genre = st.sidebar.selectbox("Choose a Genre", ["All"] + all_genres)

    with tab1:
        # Search UI improvement: Better input layout
        st.write("""## Enter a movie name below to get personalized movie recommendations based on similar content!""")
        
        movie_name = st.text_input("Search Movie:", placeholder="e.g. Avatar, Inception...", label_visibility="collapsed")

        if selected_genre != "All":
            data_filtered = data[data['genres'].apply(lambda x: selected_genre in x)]
            recommended_titles, recommended_posters_links = recommend(movie_name, data_filtered, model, similarity_index, selected_genre)
        else:
            recommended_titles, recommended_posters_links = recommend(movie_name, data, model, similarity_index, selected_genre)

        if recommended_titles:
            st.write("### Top 20 Recommended Movies")

            # Organize recommendations in a grid (3 movies per row)
            num_per_row = 3
            total_movies = len(recommended_titles)

            for start_idx in range(0, total_movies, num_per_row):
                cols = st.columns(num_per_row)  # Create columns for the current row
                for idx, col in enumerate(cols):
                    movie_idx = start_idx + idx
                    if movie_idx < total_movies:
                        title = recommended_titles[movie_idx]
                        poster, link = recommended_posters_links[movie_idx]
                        rating = data['vote_average'].iloc[movie_idx]
                        genres = ", ".join(data['genres'].iloc[movie_idx][:2])  # Show only the top two genres
                        
                        with col:
                            # Show movie details (Rating and Genre) above the poster
                            st.markdown(f"**Rating**: {rating}/10")
                            st.markdown(f"**Genres**: {genres}")
                            if poster:  # Display poster with clickable link
                                st.markdown(
                                    f"<a href='{link}' target='_blank'><img src='{poster}' style='width:100%; border-radius:10px;' alt='{title}'/></a>",
                                    unsafe_allow_html=True,
                                )
                            else:  # Placeholder for missing posters
                                st.markdown(f"**No poster available** for {title}")
    with tab2:
        # All Movies
        st.write("## All Movies")

        # Filter data based on genre selection
        if selected_genre != "All":
            data_filtered = data[data['genres'].apply(lambda x: selected_genre in x)]
        else:
            data_filtered = data

        # Sort movies by rating (highest to lowest)
        data_filtered = data_filtered.sort_values(by='vote_average', ascending=False)

        # Pagination logic: Load 100 movies at a time
        if 'page' not in st.session_state:
            st.session_state.page = 0  # Initialize page number

        # Compute the start and end indices for the current page
        start_idx = st.session_state.page * 100
        end_idx = (st.session_state.page + 1) * 100
        top_rated_movies = data_filtered.iloc[start_idx:end_idx]

        # Show in 3 columns per row
        num_per_row = 3
        total_movies = len(top_rated_movies)

        # Create a grid of movies (3 movies per row)
        for row_start_idx in range(0, total_movies, num_per_row):
            cols = st.columns(num_per_row)  # Create columns for the current row
            for col_idx, col in enumerate(cols):
                movie_idx = row_start_idx + col_idx
                if movie_idx < total_movies:
                    movie = top_rated_movies.iloc[movie_idx]
                    title = movie['title']
                    poster, link = fetch_poster(movie['id'])
                    rating = movie['vote_average']
                    genres = ", ".join(movie['genres'][:2])  # Show only top 2 genres

                    with col:
                        st.markdown(f"**Rating**: {rating}/10")
                        st.markdown(f"**Genres**: {genres}")
                        if poster:  # Display poster with clickable link
                            st.markdown(
                                f"<a href='{link}' target='_blank'><img src='{poster}' style='width:100%; border-radius:10px;' alt='{title}'/></a>",
                                unsafe_allow_html=True,
                            )
                        else:  # Placeholder for missing posters
                            st.markdown(f"**No poster available** for {title}")

        # Add button to load the next set of movies (pagination)
        if end_idx < len(data_filtered):
            if st.button("See More Movies"):
                st.session_state.page += 1  # Increment the page number

   

if __name__ == "__main__":
    main()
