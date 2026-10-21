"""Flask web frontend for the MovieLens preference analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from movie_preference_app.analysis import run_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

app = Flask(__name__)

# Cache the last successful run so repeated views don't re-cluster.
_state: dict[str, object] = {}


def _run_or_cached(
    ratings_path: Path,
    movies_path: Path,
    n_clusters: int,
    user_id: int | None,
    top_n: int,
) -> dict[str, object]:
    """Re-run the analysis only when the inputs change."""
    key = (ratings_path, movies_path, n_clusters, user_id, top_n)
    if _state.get("key") == key:
        return _state["result"]
    result = run_analysis(
        ratings_path,
        movies_path,
        OUTPUT_DIR,
        n_clusters=n_clusters,
        user_id=user_id,
        top_n=top_n,
    )
    _state["key"] = key
    _state["result"] = result
    return result


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    """Render the form; on POST run the analysis and show results."""
    form_clusters = int(request.form.get("clusters", 4))
    form_user_id = request.form.get("user_id") or None
    form_user_id = int(form_user_id) if form_user_id else None
    form_top_n = int(request.form.get("top_n", 10))

    ratings_path = PROJECT_ROOT / "ratings.csv"
    movies_path = PROJECT_ROOT / "movies.csv"

    context: dict[str, object] = {
        "clusters": form_clusters,
        "user_id": form_user_id,
        "top_n": form_top_n,
        "error": None,
        "ran": False,
    }
    if request.method == "POST":
        try:
            result = _run_or_cached(
                ratings_path, movies_path, form_clusters, form_user_id, form_top_n
            )
            context.update(
                ran=True,
                users=result["users"],
                movies_count=result["movies"],
                ratings_count=result["ratings"],
                silhouette=result["silhouette_score"],
                user_id=result["user_id"],
                cluster_id=result["cluster_id"],
                recommendations=result["recommendations"].to_dict("records"),
                genre_summary={str(k): v for k, v in result["genre_summary"].round(2).to_dict("index").items()},
                genres=list(result["genre_summary"].columns),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            context["error"] = str(error)
    return render_template("index.html", **context)


@app.route("/outputs/<path:filename>")
def outputs(filename: str) -> Any:
    """Serve generated artifacts (charts) from the outputs directory."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/assignments")
def assignments() -> Any:
    """Cluster assignments as JSON for potential charting."""
    if not _state:
        return jsonify({"error": "Run the analysis first via the form."}), 404
    result = _state["result"]
    return jsonify(result["assignments"].to_dict("records"))


def main() -> None:
    """Entry point for the ``movie-preference-web`` console script."""
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
