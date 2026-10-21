"""Download the two JSONL files from the original GSM8K repository."""

import argparse
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        url = f"{BASE_URL}/{split}.jsonl"
        destination = args.output_dir / f"{split}.jsonl"
        with urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
