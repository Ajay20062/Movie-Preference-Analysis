"""Command-line entry point for MovieLens preference analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from movie_preference_app.analysis import run_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _at_least_two_int(value: str) -> int:
    """Argparse type: accept ints >= 2 (clusters), error out otherwise."""
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError(f"{value} must be at least 2.")
    return number


def _positive_int(value: str) -> int:
    """Argparse type: accept ints >= 1, error out otherwise."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster MovieLens users by genre preferences and recommend movies."
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=PROJECT_ROOT / "ratings.csv",
        help="Path to MovieLens ratings.csv (default: project root).",
    )
    parser.add_argument(
        "--movies",
        type=Path,
        default=PROJECT_ROOT / "movies.csv",
        help="Path to MovieLens movies.csv (default: project root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for generated CSV and PNG files.",
    )
    parser.add_argument(
        "--clusters", type=_at_least_two_int, default=4, help="Number of user clusters."
    )
    parser.add_argument("--user-id", type=int, help="User to generate recommendations for.")
    parser.add_argument("--top-n", type=_positive_int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--web", action="store_true", help="Launch the Flask web frontend instead."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.web:
        from movie_preference_app.webapp import main as web_main

        web_main()
        return 0
    try:
        result = run_analysis(
            args.ratings,
            args.movies,
            args.output_dir,
            n_clusters=args.clusters,
            user_id=args.user_id,
            top_n=args.top_n,
        )
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(2, f"Error: {error}\n")

    score = result["silhouette_score"]
    print(f"Users: {result['users']:,} | Movies: {result['movies']:,} | "
          f"Ratings: {result['ratings']:,}")
    print(
        f"Silhouette score: {score:.3f}"
        if score is not None
        else "Silhouette score unavailable for this clustering."
    )
    print("\nAverage genre ratings by cluster:")
    print(result["genre_summary"].round(2).to_string())
    print(f"\nRecommendations for user {result['user_id']} "
          f"(cluster {result['cluster_id']}):")
    recommendations = result["recommendations"]
    if recommendations.empty:
        print("No highly rated unseen movies from other users in this cluster.")
    else:
        print(recommendations.to_string(index=False))
    print(f"\nOutputs saved to: {Path(result['output_dir']).resolve()}")
    return 0
