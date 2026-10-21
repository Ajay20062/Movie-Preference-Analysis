# Movie Preference Analysis

Analyze the [MovieLens Latest Small dataset](https://grouplens.org/datasets/movielens/latest/)
to discover groups of users with similar genre preferences and generate
cluster-based movie recommendations.

## What it does

1. Loads and validates `ratings.csv` and `movies.csv`.
2. Builds one average-rating feature per genre for each user.
3. Standardizes genre profiles and clusters users with K-Means.
4. Uses PCA to create a two-dimensional cluster visualization.
5. Reports the Silhouette Score and average genre preferences per cluster.
6. Recommends highly rated movies from other users in the same cluster that the
   selected user has not rated.

K-Means runs on the full standardized genre profiles; PCA is used for
visualization rather than discarding features before clustering.

## Requirements and installation

- Python 3.9 or newer
- MovieLens Latest Small `ratings.csv` and `movies.csv`

Create a virtual environment and install the project:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Download and extract the dataset. Put `ratings.csv` and `movies.csv` in the
project root, then run:

```powershell
python movie_preference.py
```

The installed command is also available:

```powershell
movie-preference --clusters 4 --user-id 1 --top-n 10
```

The defaults read the CSV files from the project root and write outputs to
`outputs/`. To override paths:

```powershell
python movie_preference.py --ratings data\ratings.csv --movies data\movies.csv --output-dir outputs --clusters 4 --user-id 1 --top-n 10
```

If `--user-id` is omitted, the most active user with a valid genre profile is
selected. Recommendations are based on ratings of at least 4.0 from *other*
users in that user's cluster.

## Generated files

- `outputs/movie_clusters.png` — PCA scatter plot, colored by K-Means cluster
- `outputs/user_clusters.csv` — each profiled user's cluster and PCA coordinates
- `outputs/sample_recommendations.csv` — ranked, unseen movie recommendations

The Silhouette Score is omitted when clustering produces too few or too many
distinct labels to calculate it meaningfully. Cluster numbers are arbitrary
labels, not preference rankings.

## Web frontend

A small Flask app serves the same analysis in the browser:

```powershell
python run_webapp.py
```

Then open http://127.0.0.1:5000. Enter the cluster count, user ID (leave blank
for the most active user) and number of recommendations, and click **Run
Analysis**. The page shows summary statistics, per-cluster genre profiles, the
recommendation list and the PCA cluster chart. Results are cached per option
set, and `GET /api/assignments` returns the cluster assignments as JSON.

## Tests

Run the test suite from the project root:

```powershell
python -m pytest
```

Tests use a small synthetic dataset and do not require downloading MovieLens.

## Limitations

Genre means can hide variation among individual movies, sparse ratings make
profiles less reliable, and K-Means requires a chosen cluster count. Cluster
recommendations are a simple baseline rather than a replacement for
collaborative-filtering recommenders.
