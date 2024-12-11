# 🎬 Movie Recommendation System

This project is a **Content-Based Movie Recommendation System** built using **Streamlit** for the frontend. It uses **TMDB (The Movie Database) API** for fetching movie posters and additional data and employs **Sentence-BERT** for generating semantic embeddings to identify similar movies based on their content. A **FAISS index** ensures efficient similarity searches.

---

## 🛠 Features

- **Personalized Movie Recommendations**: Search for a movie and get top 20 similar movies based on genres, keywords, cast, and overview.
- **Genre-Based Filtering**: Refine recommendations by selecting specific genres.
- **Interactive Movie Grid**: Browse movies with their posters, ratings, and genres.
- **Efficient Search with FAISS**: Leverages FAISS for fast nearest neighbor search in embedding space.
- **Pagination for All Movies**: Browse through the complete dataset with genre filtering.

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or later
- API key from [TMDB](https://www.themoviedb.org/documentation/api)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your TMDB API key:
   Replace `API_KEY` in the code with your TMDB API key.

4. Ensure the dataset files `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are placed in the appropriate directory.

---

## 📊 Dataset
- **tmdb_5000_movies.csv**: Contains movie metadata like title, overview, genres, keywords, etc.
- **tmdb_5000_credits.csv**: Includes cast and crew details.

---

## 🏗 Architecture
1. **Preprocessing**:
   - Merge `movies` and `credits` datasets.
   - Extract and clean data for genres, keywords, cast, and crew.
   - Combine features into a single text field for embedding.

2. **Embedding**:
   - Generate embeddings using `Sentence-BERT` for each movie's feature-rich text.

3. **Similarity Search**:
   - Build a **FAISS index** for fast nearest-neighbor search.
   - Use cosine similarity to find the most similar movies.

4. **Streamlit UI**:
   - Two tabs: `Recommendation` and `All Movies`.
   - Genre filtering and movie search input.

---

## 🖥️ Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Access the app at `http://localhost:8501` in your browser.

3. Use the **Recommendation** tab to search for movies and view recommendations. Use the **All Movies** tab to browse through all available movies.

---

## ⚡ Technologies Used
- **Python**
- **Streamlit**: Frontend framework
- **Sentence-BERT**: Generating semantic embeddings
- **FAISS**: Fast similarity search
- **Pandas**: Data manipulation
- **Requests**: API calls to TMDB

---

## 🛡️ License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🌟 Acknowledgments
- The Movie Database (TMDB) for their API and dataset.
- Sentence-Transformers library for efficient embeddings.
- FAISS for fast nearest neighbor searches.

Feel free to contribute or raise issues for enhancements!
