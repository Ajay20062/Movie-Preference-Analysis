"""MovieLens user-preference clustering and recommendation."""

from movie_preference_app.analysis import (
    make_genre_profiles,
    make_recommendations,
    run_analysis,
)

__all__ = ["make_genre_profiles", "make_recommendations", "run_analysis"]
