# Moviefy — Content‑Based Movie Recommendation System 🎬

Discover and explore movies powered by NLP and vector similarity. Moviefy analyzes genres, keywords, and overviews to recommend similar titles with clean visuals and a responsive UI.

## Live Demo
Experience it here: [`moviefy.streamlit.app`](https://moviefy.streamlit.app/)

## Features
- 🔎 Intelligent search with autocomplete (select from known titles)
- 🧠 Content-based recommendations using TF‑IDF + cosine similarity
- 📈 Match scores to explain recommendations
- 🖼️ Poster fetching via TMDB (when API key is provided)
- 📺 Netflix‑inspired dark UI (Streamlit custom CSS)
- 🗂️ Watchlist (session-based) and basic user ratings (session-based)
- 📊 Stats page with quick visual insights

## How It Works
We build a TF‑IDF representation over combined content features:
- `genres` + `keywords` + `overview`

Then we compute pairwise cosine similarity and rank movies closest to the selected title. TMDB is used to fetch posters and enrich results when API keys are configured.

## Tech Stack
- Python 3.8+
- Streamlit
- scikit-learn (TF‑IDF, cosine similarity)
- Pandas, NumPy
- Plotly (charts)
- Requests (TMDB integration)

## Getting Started (Local)

### Prerequisites
- Python 3.8+
- pip

### 1) Clone the repo
```bash
git clone https://github.com/MrKunalSharma/movie_recommendation_project.git
cd movie_recommendation
```

### 2) Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) (Optional) Configure API keys for posters/trailers
Moviefy supports TMDB (and optional YouTube) for richer visuals. In Streamlit, these are read from `secrets.toml`.

Create `.streamlit/secrets.toml`:
```toml
[general]

[secrets]
TMDB_API_KEY = "your_tmdb_api_key"
YOUTUBE_API_KEY = "your_youtube_api_key"
```

### 5) Run locally
```bash
streamlit run app.py
```

Open the provided URL in your browser.

## Dataset
Place your dataset at `data/movies.csv`. Expected columns (minimum):
- `title` (movie title)
- `genres` (pipe/comma‑separated or list‑like string)
- `keywords`
- `overview`
- `release_date` (for year extraction)
- `tmdb_id` (optional but helpful for poster fetching)

You can adapt `app.py` to your dataset schema if needed.

## Project Structure
```text
movie_recommendation/
├─ app.py
├─ data/
│  └─ movies.csv
├─ requirements.txt
├─ PROBLEM_STATEMENT.md
└─ README.md
```

## Key Screens and UX
- Home: pick a movie and see top‑N similar titles with match scores and posters.
- Watchlist: maintain a simple session‑based watchlist.
- Statistics: quick charts over the dataset (e.g., releases by year, popular genres).

## Roadmap
- ✅ Content‑based recommendations (TF‑IDF + cosine similarity)
- ✅ TMDB poster enrichment
- 🚧 Better fuzzy matching and multi‑field search
- 🚧 Hybrid recommendations (content + collaborative signals)
- 🚧 Persisted user profiles and watchlists
- 🚧 Advanced filters (rating, year, genre intersections)

## Contributing
Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Commit with clear messages
4. Open a PR describing the change and rationale

## Acknowledgements
- The Movie Database (TMDB) for posters and metadata
- Streamlit for rapid app development
- scikit‑learn for vectorization and similarity

## License
This project is licensed under the MIT License. See `LICENSE` if present, or include one in your fork.

---

Questions or suggestions? Open an issue or reach out via GitHub.
