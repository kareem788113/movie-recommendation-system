# 🎬 Movie Recommendation System
### Multi-Perspective Analysis using Pattern Mining, Graph Analytics, and BERT

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Analytics-FF8C00?style=flat-square)](https://networkx.org/)
[![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-NLP_Embeddings-FFCA28?style=flat-square)](https://www.sbert.net/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML_Toolkit-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## 🧭 Overview

Finding a good movie to watch should be simple. In practice, it rarely is. Streaming platforms offer tens of thousands of titles, and the gap between what's visible and what's actually relevant to any given person is enormous. The browsing experience breaks down — users scroll, hesitate, and often settle for something familiar rather than discovering something genuinely worth watching.

Most recommendation systems try to solve this with a single approach: collaborative filtering captures what other users with similar tastes have watched, while content-based filtering matches movies by genre or metadata. Both are useful, and both have real limits. Collaborative filtering is blind to new content and thin user histories. Content-based filtering tends to trap users inside the genres they already know, surfacing more of what they've already seen rather than something they haven't found yet.

This project takes a different angle. Instead of picking one method and tuning it, it brings three independent perspectives together — **behavioral** (what users actually watch together), **structural** (which movies occupy influential positions in a co-viewing network), and **semantic** (which movies share similar narratives at a description level). Each perspective captures something the other two miss. The goal is not only to recommend movies, but to make the reasoning behind each recommendation transparent and defensible.

> *"This project focuses not only on recommending movies, but on explaining the reasoning behind each recommendation."*

---

## 🧱 System Architecture

The pipeline is modular by design — each stage is independent, writes its output to disk, and feeds cleanly into the next. This makes it straightforward to re-run any single module without reprocessing the entire chain.

```
Raw Data  →  Data Cleaning  →  Feature Engineering  →  Pattern Mining
                                                               ↓
          Visualization  ←  Hybrid Scoring  ←  Graph Analytics + NLP Embeddings
```

| Stage | Description |
|---|---|
| **Raw Data Ingestion** | Load MovieLens interaction logs and TMDb movie metadata |
| **Data Cleaning** | Resolve missing values, remove duplicates, standardize types and formats |
| **Feature Engineering** | Build transaction arrays, graph edge lists, and normalized text fields |
| **Pattern Mining** | Run FP-Growth on user transaction data to extract association rules |
| **Graph Construction** | Build a weighted co-watching network; apply PageRank and HITS scoring |
| **NLP Embeddings** | Encode movie descriptions via Sentence-BERT; compute cosine similarity matrix |
| **Hybrid Scoring** | Combine behavioral, structural, and semantic scores into a single ranked output |
| **Visualization** | Export interpretable charts for confidence, lift, support, and top rules |

---

## 📊 Data Sources

The system draws on two complementary datasets. Neither is sufficient on its own — together they cover both the behavioral and content dimensions of a recommendation problem.

### MovieLens Dataset
Maintained by the GroupLens research lab at the University of Minnesota, MovieLens is one of the most widely cited benchmarks in recommendation system research. It provides:
- **User–movie ratings** (0.5 to 5.0 in half-star increments)
- **Interaction timestamps** — useful for sequencing and recency analysis
- **Genre labels** (pipe-delimited, per movie)

Any rating event is treated as evidence of a watch. The rating value itself is used only to filter low-confidence interactions (ratings below 3.0 are excluded from the transaction set).

### TMDb 5000 Movies Dataset
The Movie Database adds the content layer that behavioral data alone cannot provide:
- **Plot overviews** — the primary input for BERT embeddings
- **Rich genre metadata** (JSON-encoded, more granular than MovieLens labels)
- **Popularity, vote average, vote count** — used to calibrate support thresholds
- **Release date, runtime, budget, revenue** — contextual signals for exploratory analysis

Combining both datasets is what makes the hybrid approach possible. MovieLens tells us *who watches what*. TMDb tells us *what each movie is actually about*. Cross-referencing the two gives a complete picture that neither source provides alone.

---

## ⚙️ Data Engineering Pipeline

> *"Reliable recommendations start with reliable data."*

Raw data from both sources arrived with inconsistencies that had to be resolved before any analysis could be trusted. The cleaning pipeline (`01_data_preprocessing.ipynb`) handles all of this and exports structured outputs to `data/processed/`.

### Data Cleaning

- **Missing values:** Critical fields (`userId`, `movieId`, `title`) were dropped if null — these cannot be imputed meaningfully. Non-critical fields (`runtime`, `budget`) were retained with null markers since they aren't used in computation.
- **Duplicates:** Repeated movie records inflate support counts; repeated user–movie interaction pairs add false weight to graph edges. Both were collapsed to single canonical records.
- **Type standardization:** `movieId` was cast to integer across both datasets. Column names were normalized to `snake_case`. The TMDb `genres` field was parsed from JSON into a plain comma-separated string.
- **Text cleaning:** Overview text was lowercased, Unicode-normalized, and stripped of HTML artifacts. No stemming or stop-word removal — Sentence-BERT performs better on grammatically complete sentences.

### Data Transformation

- **Transaction dataset:** Ratings grouped by `userId`. Each user's complete set of watched movie titles (rating ≥ 3.0) becomes one transaction row in `transactions.csv`.
- **Graph edge list:** All ordered pairs of movies in a shared watch history become directed edges. Edge weight = number of distinct users who watched both films.
- **Text corpus:** Cleaned `overview` fields assembled per movie for NLP embedding.

### Data Integration

MovieLens ratings were left-joined with MovieLens movie metadata on `movieId`, then joined with the TMDb dataset via a cross-referenced ID mapping. The result — `cleaned_dataset.csv` — carries behavioral signals (`userId`, `movieId`, `rating`) and content signals (`title`, `genres`, `overview`, `popularity`) in a single file used across all analytical modules.

---

## 🧠 Analytical Approaches

### 1. Pattern Mining — The Behavioral Perspective

Pattern mining works on a straightforward premise: if a large number of users have watched the same set of movies, that combination is probably not random. The **FP-Growth** (Frequent Pattern Growth) algorithm scans user transaction data to find which movie combinations appear together with unusual frequency, then converts those itemsets into directional association rules.

FP-Growth was chosen over Apriori for its memory efficiency — rather than generating and testing candidate itemsets iteratively, it builds a compressed prefix-tree (FP-tree) and mines it directly.

**Metrics that matter:**

| Metric | What It Measures |
|---|---|
| **Support** | How frequently this movie combination appears across all user histories |
| **Confidence** | How likely a user is to watch Movie B, given they already watched Movie A |
| **Lift** | How much stronger the relationship is compared to random chance — the most useful signal |
| **Leverage** | The raw difference between observed and expected co-occurrence — useful for sparse data |

A rule is worth acting on when confidence is high *and* lift is meaningfully above 1.0. High confidence with lift near 1.0 usually just reflects two blockbusters that everyone happens to have seen — that's popularity, not a recommendation signal. The most valuable rules tend to be mid-confidence with high lift: reliable predictions for less obvious pairings.

> *"This layer captures how users naturally consume content — the patterns that emerge not from what they say they like, but from what they actually watch."*

---

### 2. Graph Analytics — The Structural Perspective

The co-watching graph models the movie ecosystem as a weighted network. Each movie is a node. Each time two movies appear in the same user's watch history, an edge is drawn between them, with its weight incrementing by one for each additional user who shared that pair.

**PageRank** — originally designed to rank web pages by importance — is applied to identify which movies are structurally central: not the most watched, but the most *connected* in ways that matter. A movie that serves as a bridge between the thriller community and the literary drama community will score higher than its raw view count would suggest.

The **HITS** algorithm adds a second dimension: hub scores (how well a movie connects users to authoritative content) and authority scores (how dominant a movie is within its cluster). Together these surface four distinct types of content:

| Movie Type | Characteristics | Best Used For |
|---|---|---|
| Hub films | High PageRank, broad cross-genre connections | Onboarding users with no viewing history |
| Bridge films | High PageRank relative to raw views | Expanding viewing horizons across genres |
| Authority films | High HITS authority score | Deep dives within an established genre |
| Peripheral films | Low PageRank, niche audiences | Served better by semantic similarity |

> *"A movie can be influential even if it's not the most watched. PageRank finds the connective tissue — the titles that hold the network together."*

---

### 3. NLP Similarity — The Semantic Perspective

Movie descriptions carry information that user behavior simply cannot see. Two films might attract entirely different audiences historically, yet describe nearly identical narrative arcs, emotional tones, or settings. The semantic layer catches that connection.

**Sentence-BERT** (`all-MiniLM-L6-v2`) encodes each movie's plot overview into a dense 384-dimensional vector. Because SBERT uses a siamese network architecture fine-tuned for sentence-level comparison, the resulting embeddings are directly comparable by cosine distance — no need for N² separate inference passes.

The output is a full pairwise cosine similarity matrix, from which the top-K most semantically similar movies are extracted per title and stored in `top_similar_movies.csv`.

This layer is particularly valuable in two scenarios:
- **Cold start:** A new user shares one movie they enjoy — BERT generates a related list immediately, without any behavioral data.
- **Discovery:** Niche documentaries, foreign-language films, and older titles that share thematic DNA with mainstream hits, but whose audiences never historically overlapped.

> *"This helps recommend movies even when user data is limited — and often surfaces pairings that no behavioral system would ever find."*

---

## 📈 Key Insights

- Users follow clear, measurable viewing patterns. Franchise binges, director-driven sequences, and genre clusters emerge naturally from the transaction data without any manual labeling.
- High-lift association rules reveal strong hidden relationships — often between less popular titles where the audience overlap is concentrated and reliable.
- Popularity does not equal influence. Several of the highest-PageRank films in the network are not blockbusters; they are well-connected titles that serve as entry points between genre communities.
- Graph analysis consistently surfaces "bridge" movies that a pure popularity ranking would never recommend — and these tend to be the most effective at broadening a user's viewing horizon.
- Semantic similarity captures thematic connections that genre tags miss entirely. Two films can be tagged identically and feel nothing alike; two films with different tags can be thematically indistinguishable.
- Combining all three methods consistently produces more defensible, more diverse recommendations than any single signal alone — and the failure modes of each method are different enough that they genuinely complement each other.

---

## 📊 Visual Insights

Four charts are generated as part of the visualization pipeline (`05_visualization.ipynb`) to make the pattern mining results readable without opening any CSV file.

| Chart | What It Shows | Why It Matters |
|---|---|---|
| **Confidence Distribution** | Spread of confidence values across all generated rules | Reveals the proportion of actionable vs. noise rules; helps calibrate the confidence threshold |
| **Lift vs. Confidence** | Rules plotted by both metrics simultaneously | The top-right quadrant (high confidence + high lift) contains the strongest, most non-random recommendations |
| **Top Movies by Support** | Most frequently co-watched films across all transactions | Identifies behavioral anchors; useful as cold-start defaults for new users |
| **Top 10 Rules by Lift** | Ten rules with the strongest non-random association | These are the recommendations that add the most value over a plain popularity list |

---

## 🆚 Method Comparison

| Method | What It Captures | Strength | Limitation |
|---|---|---|---|
| **Pattern Mining** | User behavior — what people actually watch together | Grounded in real observed patterns; produces interpretable if-then rules | Requires sufficient interaction history; fails silently when data is sparse |
| **Graph Analytics** | Network influence — structural position in the co-watching ecosystem | Identifies bridge and hub films that popularity rankings miss | Needs a reasonably dense graph; purely structural with no semantic understanding |
| **BERT Similarity** | Narrative meaning — what films are actually about | Works without any user history; handles cold-start and cross-language comparisons | Quality depends entirely on description richness; computationally heavier |

The critical observation here is that the **failure modes don't overlap**. Pattern mining fails when data is sparse. PageRank fails when the graph is thin. BERT fails when descriptions are poor. Because the three methods go quiet in different situations, combining them covers more ground than any one of them does alone.

---

## 📁 Project Structure

```text
movie-recommendation-system/
│
├── data/
│   ├── raw/                        # Original, unmodified source files
│   │   ├── movies.csv              # MovieLens movie metadata
│   │   ├── ratings.csv             # MovieLens user–movie ratings
│   │   └── tmdb_5000_movies.csv    # TMDb movie descriptions and metadata
│   │
│   └── processed/                  # Cleaned and derived outputs
│       ├── cleaned_dataset.csv     # Merged base dataset (all modules)
│       ├── transactions.csv        # One row per user — watched movie titles
│       ├── frequent_itemsets.csv   # FP-Growth frequent itemsets
│       ├── association_rules.csv   # Scored association rules (support, confidence, lift)
│       ├── graph_data.csv          # Edge list: source, target, weight
│       ├── pagerank_scores.csv     # PageRank + HITS scores per movie
│       ├── similarity_matrix.csv   # N × N cosine similarity (BERT embeddings)
│       └── top_similar_movies.csv  # Top-K semantic matches per movie
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Load, clean, merge, export
│   ├── 02_pattern_mining.ipynb       # FP-Growth, association rules
│   ├── 03_graph_analysis.ipynb       # Graph construction, PageRank, HITS
│   ├── 04_bert_analysis.ipynb        # Sentence-BERT embeddings, cosine similarity
│   └── 05_visualization.ipynb        # All charts and visual outputs
│
├── outputs/
│   └── charts/                     # Exported visualization files
│
├── report/
│   └── final_report.pdf            # Full written report
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

### Data Processing
- **Python 3.9+** — core language
- **Pandas** — data loading, cleaning, transformation, and merging
- **NumPy** — numerical operations and matrix manipulation

### Pattern Mining
- **mlxtend** — FP-Growth algorithm, frequent itemset extraction, association rule generation

### Graph Analytics
- **NetworkX** — graph construction, PageRank, HITS, edge filtering

### NLP & Embeddings
- **Sentence-Transformers (SBERT)** — `all-MiniLM-L6-v2` for dense sentence-level embeddings
- **Scikit-learn** — pairwise cosine similarity computation

### Visualization
- **Matplotlib** — static charts and distribution plots
- **Seaborn** — styled statistical visualizations
- **Plotly** — interactive graph exploration

### Development Environment
- **Jupyter Notebook / Lab** — interactive development, inline visualization, reproducible analysis
- **Git / GitHub** — version control and project hosting

---

## ⚠️ Challenges & Limitations

**Data Sparsity.** Many users in the MovieLens dataset have rated only a handful of movies. Sparse histories produce fewer, less reliable association rules. Lowering the minimum support threshold helps recover niche patterns but introduces more noise — there's no clean answer here.

**Cold Start — New Users.** First-time users have no history. Pattern mining and graph analysis both go silent. BERT partially addresses this by generating a related list from a single named movie, but a complete cold-start solution would also need demographic signals or explicit preference selection during onboarding.

**Cold Start — New Releases.** Newly released films won't appear in the co-watching graph until sufficient interaction data accumulates. BERT handles them immediately, as long as a plot description exists at release time.

**Scalability.** At the scale of a production streaming platform, FP-Growth on the full transaction set and cosine similarity over dense embedding matrices require distributed processing (Spark, Dask) and approximate nearest-neighbor search (Faiss, ScaNN) to remain practical.

**Metadata Quality.** BERT similarity is only as good as the descriptions it reads. Short or generic overviews produce embeddings that cluster poorly. Films with detailed, specific descriptions benefit disproportionately.

**Weight Tuning.** The hybrid formula weights (40% behavioral, 30% structural, 30% semantic) are informed estimates, not optimized values. Without offline evaluation datasets and A/B test infrastructure, the optimal weights can't be determined. This is the most significant gap between the current prototype and a production deployment.

---


## 💼 Business Impact

For a streaming platform, the practical value of a better recommendation system comes down to one question: how quickly does a user find something worth watching?

A system that relies purely on popularity keeps cycling the same titles in front of everyone. A system that understands behavioral patterns, network influence, and content similarity can guide users toward titles they'll genuinely engage with — including content they would never have found by browsing.

| Benefit | Why It Matters |
|---|---|
| **Better personalization** | Recommendations reflect individual taste rather than aggregate trends |
| **Content discovery** | Users find relevant niche and catalogue titles they'd otherwise miss |
| **Reduced friction** | Less time browsing means more time watching — directly measurable in session length |
| **Higher retention** | Users who consistently find relevant content are measurably less likely to churn |
| **Long-tail monetization** | Structural and semantic methods surface quality content with low view counts that would otherwise generate no engagement |
| **Cold-start readiness** | New users get useful recommendations from day one via BERT and PageRank — no behavioral history required |
| **Transferability** | The same analytical framework applies to books, music, podcasts, or any domain with descriptions and interaction histories |


---

## ⭐ Final Thoughts

*"This project is not just about building a recommendation engine — it's about understanding how users interact with content, how movies relate to each other, and how meaning can drive better recommendations. Each method answers a different question. Together, they tell a more complete story."*

---

<div align="center">

**Movie Recommendation System** · Pattern Mining · Graph Analytics · BERT NLP

*Data Science Capstone · May 2026*

</div>
