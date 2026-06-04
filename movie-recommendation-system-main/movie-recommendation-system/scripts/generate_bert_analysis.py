import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"[^a-zA-Z0-9\s]", " ", title.lower())
    return re.sub(r"\s+", " ", title).strip()


def build_notebook() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebook_path = NOTEBOOKS_DIR / "bert_analysis.ipynb"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# BERT Analysis for Movie Recommendation\n",
                "\n",
                "This notebook implements semantic analysis of movie descriptions using Sentence-BERT.\n",
                "\n",
                "Deliverables generated:\n",
                "- `outputs/similarity_matrix.csv`\n",
                "- `outputs/top_similar_movies.csv`\n",
                "\n",
                "It also includes a comparison between content-based and behavior-based recommendations.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import re\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sentence_transformers import SentenceTransformer\n",
                "from sklearn.metrics.pairwise import cosine_similarity\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "PROJECT_ROOT = Path.cwd().resolve().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "RAW_DIR = PROJECT_ROOT / 'data' / 'raw'\n",
                "OUTPUTS_DIR = PROJECT_ROOT / 'outputs'\n",
                "OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "def clean_text(text: str) -> str:\n",
                "    text = text.lower()\n",
                "    text = re.sub(r'[^a-z0-9\\\\s]', ' ', text)\n",
                "    text = re.sub(r'\\\\s+', ' ', text).strip()\n",
                "    return text\n",
                "\n",
                "def normalize_title(title: str) -> str:\n",
                "    title = re.sub(r'\\\\(\\\\d{4}\\\\)', '', title)\n",
                "    title = re.sub(r'[^a-zA-Z0-9\\\\s]', ' ', title.lower())\n",
                "    return re.sub(r'\\\\s+', ' ', title).strip()\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1) Text preprocessing\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tmdb = pd.read_csv(RAW_DIR / 'tmdb_5000_movies.csv')\n",
                "content_df = (\n",
                "    tmdb[['title', 'overview']]\n",
                "    .dropna(subset=['title', 'overview'])\n",
                "    .drop_duplicates(subset=['title'])\n",
                "    .reset_index(drop=True)\n",
                ")\n",
                "content_df['clean_overview'] = content_df['overview'].astype(str).map(clean_text)\n",
                "content_df.head()\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2) Embedding generation (Sentence-BERT)\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = SentenceTransformer('all-MiniLM-L6-v2')\n",
                "embeddings = model.encode(\n",
                "    content_df['clean_overview'].tolist(),\n",
                "    batch_size=32,\n",
                "    show_progress_bar=True,\n",
                "    convert_to_numpy=True,\n",
                ")\n",
                "embeddings.shape\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3) Similarity computation (cosine)\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "content_similarity = cosine_similarity(embeddings)\n",
                "\n",
                "# Save matrix for report (first 500 movies keeps file size practical)\n",
                "matrix_size = min(500, len(content_df))\n",
                "matrix_titles = content_df['title'].iloc[:matrix_size]\n",
                "similarity_matrix_df = pd.DataFrame(\n",
                "    content_similarity[:matrix_size, :matrix_size],\n",
                "    index=matrix_titles,\n",
                "    columns=matrix_titles,\n",
                ")\n",
                "similarity_matrix_df.to_csv(OUTPUTS_DIR / 'similarity_matrix.csv', index=True)\n",
                "similarity_matrix_df.shape\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4) Top similar movies (content-based)\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "top_k = 10\n",
                "rows = []\n",
                "titles = content_df['title'].tolist()\n",
                "\n",
                "for i, movie_title in enumerate(titles):\n",
                "    sims = content_similarity[i]\n",
                "    top_indices = np.argsort(sims)[::-1]\n",
                "    rank = 0\n",
                "    for j in top_indices:\n",
                "        if i == j:\n",
                "            continue\n",
                "        rank += 1\n",
                "        rows.append({\n",
                "            'movie_title': movie_title,\n",
                "            'similar_movie_title': titles[j],\n",
                "            'similarity_score': float(sims[j]),\n",
                "            'rank': rank,\n",
                "        })\n",
                "        if rank >= top_k:\n",
                "            break\n",
                "\n",
                "top_similar_df = pd.DataFrame(rows)\n",
                "top_similar_df.to_csv(OUTPUTS_DIR / 'top_similar_movies.csv', index=False)\n",
                "top_similar_df.head(20)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5) Comparative analysis: content-based vs behavior-based\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "movies = pd.read_csv(RAW_DIR / 'movies.csv')\n",
                "ratings = pd.read_csv(RAW_DIR / 'ratings.csv')\n",
                "\n",
                "# Implicit behavior matrix from ratings (liked = rating >= 4)\n",
                "ratings['liked'] = (ratings['rating'] >= 4.0).astype(int)\n",
                "user_item = ratings.pivot_table(index='userId', columns='movieId', values='liked', fill_value=0)\n",
                "item_user = user_item.T\n",
                "behavior_similarity = cosine_similarity(item_user)\n",
                "behavior_movie_ids = item_user.index.to_numpy()\n",
                "\n",
                "behavior_titles = movies.set_index('movieId')['title']\n",
                "behavior_norm = {mid: normalize_title(behavior_titles.get(mid, '')) for mid in behavior_movie_ids}\n",
                "content_norm_to_idx = {normalize_title(t): i for i, t in enumerate(content_df['title'])}\n",
                "\n",
                "# Find overlap and choose a strong seed movie with enough behavioral interactions\n",
                "ratings_count = ratings.groupby('movieId').size().to_dict()\n",
                "overlap = []\n",
                "for pos, mid in enumerate(behavior_movie_ids):\n",
                "    n = behavior_norm.get(mid, '')\n",
                "    if n in content_norm_to_idx and n:\n",
                "        overlap.append((pos, mid, n, ratings_count.get(mid, 0)))\n",
                "\n",
                "seed_pos, seed_mid, seed_norm, _ = sorted(overlap, key=lambda x: x[3], reverse=True)[0]\n",
                "seed_content_idx = content_norm_to_idx[seed_norm]\n",
                "\n",
                "def top_content(seed_idx, k=5):\n",
                "    sims = content_similarity[seed_idx]\n",
                "    ids = np.argsort(sims)[::-1]\n",
                "    out = []\n",
                "    for j in ids:\n",
                "        if j == seed_idx:\n",
                "            continue\n",
                "        out.append((content_df.iloc[j]['title'], float(sims[j])))\n",
                "        if len(out) >= k:\n",
                "            break\n",
                "    return out\n",
                "\n",
                "def top_behavior(seed_pos_idx, k=5):\n",
                "    sims = behavior_similarity[seed_pos_idx]\n",
                "    ids = np.argsort(sims)[::-1]\n",
                "    out = []\n",
                "    for j in ids:\n",
                "        if j == seed_pos_idx:\n",
                "            continue\n",
                "        mid = behavior_movie_ids[j]\n",
                "        out.append((behavior_titles.get(mid, f'movieId={mid}'), float(sims[j])))\n",
                "        if len(out) >= k:\n",
                "            break\n",
                "    return out\n",
                "\n",
                "seed_title_content = content_df.iloc[seed_content_idx]['title']\n",
                "seed_title_behavior = behavior_titles.get(seed_mid)\n",
                "\n",
                "content_top = top_content(seed_content_idx, k=5)\n",
                "behavior_top = top_behavior(seed_pos, k=5)\n",
                "\n",
                "comparison_df = pd.DataFrame({\n",
                "    'rank': [1, 2, 3, 4, 5],\n",
                "    'content_based_recommendation': [x[0] for x in content_top],\n",
                "    'content_similarity': [x[1] for x in content_top],\n",
                "    'behavior_based_recommendation': [x[0] for x in behavior_top],\n",
                "    'behavior_similarity': [x[1] for x in behavior_top],\n",
                "})\n",
                "\n",
                "print('Seed (content title):', seed_title_content)\n",
                "print('Seed (behavior title):', seed_title_behavior)\n",
                "comparison_df\n",
            ],
        },
    ]

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path.write_text(json.dumps(notebook_data, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    tmdb = pd.read_csv(RAW_DIR / "tmdb_5000_movies.csv")
    content_df = (
        tmdb[["title", "overview"]]
        .dropna(subset=["title", "overview"])
        .drop_duplicates(subset=["title"])
        .reset_index(drop=True)
    )
    content_df["clean_overview"] = content_df["overview"].astype(str).map(clean_text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        content_df["clean_overview"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    content_similarity = cosine_similarity(embeddings)

    # Keep matrix export practical for CSV size.
    matrix_size = min(500, len(content_df))
    matrix_titles = content_df["title"].iloc[:matrix_size]
    similarity_matrix_df = pd.DataFrame(
        content_similarity[:matrix_size, :matrix_size],
        index=matrix_titles,
        columns=matrix_titles,
    )
    similarity_matrix_df.to_csv(OUTPUTS_DIR / "similarity_matrix.csv", index=True)

    top_k = 10
    rows = []
    titles = content_df["title"].tolist()
    for i, movie_title in enumerate(titles):
        sims = content_similarity[i]
        top_indices = np.argsort(sims)[::-1]
        rank = 0
        for j in top_indices:
            if i == j:
                continue
            rank += 1
            rows.append(
                {
                    "movie_title": movie_title,
                    "similar_movie_title": titles[j],
                    "similarity_score": float(sims[j]),
                    "rank": rank,
                }
            )
            if rank >= top_k:
                break

    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "top_similar_movies.csv", index=False)
    build_notebook()

    print("Generated:")
    print(f"- {NOTEBOOKS_DIR / 'bert_analysis.ipynb'}")
    print(f"- {OUTPUTS_DIR / 'similarity_matrix.csv'}")
    print(f"- {OUTPUTS_DIR / 'top_similar_movies.csv'}")


if __name__ == "__main__":
    main()
