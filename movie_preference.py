"""Compatibility launcher for the MovieLens preference analysis."""

import sys
from pathlib import Path

# Allow running directly from a checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from movie_preference_app.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
