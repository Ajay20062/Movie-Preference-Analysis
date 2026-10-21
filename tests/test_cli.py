import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from movie_preference_app.cli import build_parser, main


class CliParserTests(unittest.TestCase):
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.clusters, 4)
        self.assertEqual(args.top_n, 10)
        self.assertIsNone(args.user_id)
        self.assertEqual(args.output_dir, Path.cwd() / "outputs")
        self.assertEqual(args.ratings, Path.cwd() / "ratings.csv")
        self.assertEqual(args.movies, Path.cwd() / "movies.csv")

    def test_rejects_fewer_than_two_clusters(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--clusters", "1"])

    def test_rejects_non_positive_top_n(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--top-n", "0"])

    def test_accepts_valid_arguments(self):
        args = build_parser().parse_args(
            ["--clusters", "3", "--top-n", "5", "--user-id", "7"]
        )
        self.assertEqual((args.clusters, args.top_n, args.user_id), (3, 5, 7))


class CliMainTests(unittest.TestCase):
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

    def test_main_returns_zero_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ratings_path = root / "ratings.csv"
            movies_path = root / "movies.csv"
            output_dir = root / "outputs"
            self.ratings.to_csv(ratings_path, index=False)
            self.movies.to_csv(movies_path, index=False)
            argv = [
                "--ratings",
                str(ratings_path),
                "--movies",
                str(movies_path),
                "--output-dir",
                str(output_dir),
                "--clusters",
                "2",
                "--user-id",
                "1",
                "--top-n",
                "3",
            ]
            with mock.patch("builtins.print"):
                self.assertEqual(main(argv), 0)
            for filename in (
                "user_clusters.csv",
                "sample_recommendations.csv",
                "movie_clusters.png",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)

    def test_main_exits_with_error_on_missing_files(self):
        parser_exit = mock.patch(
            "argparse.ArgumentParser.exit",
            side_effect=SystemExit(2),
        )
        with parser_exit, mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as caught:
                main(["--ratings", "missing.csv", "--movies", "missing.csv"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
