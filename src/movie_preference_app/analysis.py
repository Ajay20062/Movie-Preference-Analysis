"""Core MovieLens data loading, clustering, visualization, and recommendation logic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RATING_THRESHOLD = 4.0
RANDOM_STATE = 42


def load_data(
    ratings_path: Path, movies_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens CSVs and reject malformed or unusable input."""
    for path in (ratings_path, movies_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {path}. Download MovieLens Latest Small "
                "and provide both ratings.csv and movies.csv."
            )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    required_ratings = {"userId", "movieId", "rating"}
    required_movies = {"movieId", "title", "genres"}
    missing_ratings = required_ratings - set(ratings.columns)
    missing_movies = required_movies - set(movies.columns)
    if missing_ratings:
        raise ValueError(f"{ratings_path} is missing columns: {sorted(missing_ratings)}")
    if missing_movies:
        raise ValueError(f"{movies_path} is missing columns: {sorted(missing_movies)}")

    ratings = ratings.loc[:, ["userId", "movieId", "rating"]].copy()
    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings = ratings[
        np.isfinite(ratings["userId"])
        & np.isfinite(ratings["movieId"])
        & np.isfinite(ratings["rating"])
    ]
    ratings = ratings[
        (ratings["userId"] > 0)
        & (ratings["movieId"] > 0)
        & ratings["rating"].between(0.5, 5.0)
    ]
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last")
    if ratings.empty:
        raise ValueError("The ratings file contains no valid ratings.")

    movies = movies.loc[:, ["movieId", "title", "genres"]].copy()
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId", "title", "genres"])
    movies = movies[np.isfinite(movies["movieId"]) & (movies["movieId"] > 0)]
    movies["movieId"] = movies["movieId"].astype(int)
    movies = movies.drop_duplicates("movieId", keep="first")
    if movies.empty:
        raise ValueError("The movies file contains no valid movie records.")
    return ratings, movies


def make_genre_profiles(
    ratings: pd.DataFrame, movies: pd.DataFrame
) -> pd.DataFrame:
    """Build user-by-genre mean-rating profiles and impute missing genres."""
    joined = ratings.merge(
        movies[["movieId", "genres"]].drop_duplicates("movieId"),
        on="movieId",
        how="inner",
        validate="many_to_one",
    )
    joined["genre"] = joined["genres"].astype(str).str.split("|")
    exploded = joined.explode("genre")
    exploded = exploded[
        exploded["genre"].notna()
        & (exploded["genre"] != "")
        & (exploded["genre"] != "(no genres listed)")
    ]
    if exploded.empty:
        raise ValueError("No rated movies with usable genres were found in the data.")

    profiles = exploded.pivot_table(
        index="userId", columns="genre", values="rating", aggfunc="mean"
    )
    profiles = profiles.fillna(profiles.mean()).dropna(axis="columns", how="all")
    profiles.columns.name = None
    if profiles.empty or profiles.shape[1] == 0:
        raise ValueError("Could not create any user genre preference features.")
    return profiles.sort_index()


def make_recommendations(
    user_id: int,
    cluster_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    assignments: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank unseen movies rated highly by other users in the same cluster."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    cluster_users = assignments.loc[
        (assignments["cluster"] == cluster_id)
        & (assignments["userId"] != user_id),
        "userId",
    ]
    watched = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    candidates = ratings[
        ratings["userId"].isin(cluster_users)
        & (ratings["rating"] >= RATING_THRESHOLD)
        & ~ratings["movieId"].isin(watched)
    ]
    ranked = (
        candidates.groupby("movieId")
        .agg(
            cluster_average_rating=("rating", "mean"),
            cluster_rating_count=("rating", "count"),
        )
        .reset_index()
        .sort_values(
            ["cluster_average_rating", "cluster_rating_count", "movieId"],
            ascending=[False, False, True],
        )
        .head(top_n)
    )
    return ranked.merge(
        movies[["movieId", "title", "genres"]].drop_duplicates("movieId"),
        on="movieId",
        how="left",
        validate="one_to_one",
    )[
        [
            "movieId",
            "title",
            "genres",
            "cluster_average_rating",
            "cluster_rating_count",
        ]
    ]


def _pca_coordinates(scaled_profiles: np.ndarray) -> tuple[np.ndarray, PCA]:
    components = min(2, scaled_profiles.shape[0], scaled_profiles.shape[1])
    pca = PCA(n_components=components, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(scaled_profiles)
    if components == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
    return coordinates, pca


def _cluster_profiles(
    profiles: pd.DataFrame, n_clusters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA, float | None]:
    """Standardize profiles, cluster users, and compute PCA coordinates."""
    scaled = StandardScaler().fit_transform(profiles)
    # Clustering uses the full standardized preference vectors; PCA is for display.
    labels = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    ).fit_predict(scaled)
    coordinates, pca = _pca_coordinates(scaled)

    unique_labels = np.unique(labels)
    score = (
        float(silhouette_score(scaled, labels))
        if 2 <= len(unique_labels) < len(profiles)
        else None
    )
    return scaled, labels, coordinates, pca, score


def _plot_clusters(
    output_dir: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    pca: PCA,
    highlight_user: int | None = None,
    highlight_coordinate: np.ndarray | None = None,
) -> None:
    """Save a polished PCA scatter plot of user clusters to movie_clusters.png."""
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor("white")

    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
    for cluster_id in np.unique(labels):
        points = coordinates[labels == cluster_id]
        color = colors[cluster_id % len(colors)]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=34,
            color=color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
            label=f"Cluster {cluster_id} ({len(points)} users)",
            zorder=3,
        )
        centroid = points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=260,
            marker="X",
            color=color,
            edgecolors="#1e2749",
            linewidths=1.6,
            zorder=4,
        )

    if highlight_coordinate is not None:
        ax.scatter(
            highlight_coordinate[0],
            highlight_coordinate[1],
            s=220,
            facecolors="none",
            edgecolors="#1e2749",
            linewidths=2.0,
            zorder=5,
        )
        ax.annotate(
            f"user {highlight_user}",
            (highlight_coordinate[0], highlight_coordinate[1]),
            textcoords="offset points",
            xytext=(10, 8),
            fontsize=10,
            fontweight="bold",
            color="#1e2749",
            zorder=5,
        )

    explained = pca.explained_variance_ratio_
    xlabel = f"Principal Component 1 ({explained[0]:.1%} variance explained)"
    ylabel = (
        f"Principal Component 2 ({explained[1]:.1%} variance explained)"
        if len(explained) > 1
        else "Principal Component 2"
    )
    ax.set_title(
        "MovieLens User Clusters (PCA)",
        fontsize=15,
        fontweight="bold",
        color="#1e2749",
        pad=14,
    )
    ax.set_xlabel(xlabel, fontsize=11, color="#1e2749")
    ax.set_ylabel(ylabel, fontsize=11, color="#1e2749")
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#4b5563")
    legend = ax.legend(
        title="Cluster",
        loc="best",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        title_fontsize=10,
    )
    legend.get_title().set_fontweight("bold")
    fig.text(
        0.99,
        0.01,
        "✕ = cluster centroid",
        ha="right",
        fontsize=8.5,
        color="#6b7280",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(output_dir / "movie_clusters.png", dpi=150)
    plt.close(fig)


def run_analysis(
    ratings_path: Path,
    movies_path: Path,
    output_dir: Path,
    n_clusters: int = 4,
    user_id: int | None = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Run the analysis, save output artifacts, and return summary data."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    ratings, movies = load_data(ratings_path, movies_path)
    profiles = make_genre_profiles(ratings, movies)
    if len(profiles) < 2:
        raise ValueError("At least two users with usable genre profiles are required.")
    if not 2 <= n_clusters <= len(profiles):
        raise ValueError(f"Choose between 2 and {len(profiles)} clusters.")

    if user_id is None:
        user_id = int(ratings["userId"].value_counts().idxmax())
    if user_id not in profiles.index:
        raise ValueError(f"User {user_id} has no usable genre preference profile.")

    scaled, labels, coordinates, pca, score = _cluster_profiles(profiles, n_clusters)
    assignments = pd.DataFrame(
        {
            "userId": profiles.index,
            "cluster": labels,
            "pca_1": coordinates[:, 0],
            "pca_2": coordinates[:, 1],
        }
    )

    cluster_id = int(assignments.loc[assignments["userId"] == user_id, "cluster"].iloc[0])
    recommendations = make_recommendations(
        user_id, cluster_id, ratings, movies, assignments, top_n
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "user_clusters.csv", index=False)
    recommendations.to_csv(output_dir / "sample_recommendations.csv", index=False)
    user_point = coordinates[assignments.index[assignments["userId"] == user_id][0]]
    _plot_clusters(
        output_dir,
        coordinates,
        labels,
        pca,
        highlight_user=user_id,
        highlight_coordinate=user_point,
    )

    genre_summary = profiles.assign(cluster=labels).groupby("cluster").mean()
    return {
        "users": len(profiles),
        "movies": int(movies["movieId"].nunique()),
        "ratings": len(ratings),
        "assignments": assignments,
        "genre_summary": genre_summary,
        "silhouette_score": score,
        "recommendations": recommendations,
        "user_id": user_id,
        "cluster_id": cluster_id,
        "output_dir": output_dir,
    }
