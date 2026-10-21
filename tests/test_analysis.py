import tempfile
import unittest
from pathlib import Path

import pandas as pd

from movie_preference_app.analysis import (
    load_data,
    make_genre_profiles,
    make_recommendations,
    run_analysis,
)


class MoviePreferenceTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame(
            [
                (1, 1, 5.0),
                (1, 2, 3.0),
                (2, 1, 4.5),
                (2, 3, 5.0),
                (3, 2, 4.5),
                (3, 4, 5.0),
                (4, 3, 4.0),
                (4, 4, 4.5),
            ],
            columns=["userId", "movieId", "rating"],
        )
        self.movies = pd.DataFrame(
            [
                (1, "Drama One", "Drama"),
                (2, "Comedy One", "Comedy"),
                (3, "Action One", "Action"),
                (4, "Romance One", "Romance"),
            ],
            columns=["movieId", "title", "genres"],
        )

    def test_profiles_are_user_by_genre_and_impute_missing_genres(self):
        profiles = make_genre_profiles(self.ratings, self.movies)
        self.assertEqual(set(profiles.index), {1, 2, 3, 4})
        self.assertEqual(set(profiles.columns), {"Drama", "Comedy", "Action", "Romance"})
        self.assertFalse(profiles.isna().any().any())

    def test_recommendations_exclude_watched_movies_and_selected_user(self):
        assignments = pd.DataFrame({"userId": [1, 2, 3, 4], "cluster": [0, 0, 0, 1]})
        recommendations = make_recommendations(
            1, 0, self.ratings, self.movies, assignments, top_n=5
        )
        self.assertNotIn(1, recommendations["movieId"].tolist())
        self.assertNotIn(2, recommendations["movieId"].tolist())
        self.assertEqual(recommendations["movieId"].tolist(), [3, 4])

    def test_load_data_rejects_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            load_data(Path("missing-ratings.csv"), Path("missing-movies.csv"))

    def test_pipeline_writes_all_expected_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ratings_path = root / "ratings.csv"
            movies_path = root / "movies.csv"
            self.ratings.to_csv(ratings_path, index=False)
            self.movies.to_csv(movies_path, index=False)
            output_dir = root / "outputs"
            result = run_analysis(
                ratings_path,
                movies_path,
                output_dir,
                n_clusters=2,
                user_id=1,
                top_n=3,
            )
            self.assertEqual(result["users"], 4)
            self.assertIsNotNone(result["silhouette_score"])
            for filename in (
                "user_clusters.csv",
                "sample_recommendations.csv",
                "movie_clusters.png",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)

    def test_pipeline_rejects_invalid_cluster_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ratings_path = root / "ratings.csv"
            movies_path = root / "movies.csv"
            self.ratings.to_csv(ratings_path, index=False)
            self.movies.to_csv(movies_path, index=False)
            with self.assertRaisesRegex(ValueError, "Choose between"):
                run_analysis(ratings_path, movies_path, root / "out", n_clusters=5)


if __name__ == "__main__":
    unittest.main()
