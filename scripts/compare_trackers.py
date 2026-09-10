import argparse
import subprocess
import time
from pathlib import Path

import yaml


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_tracker(config_path: str, tracker_name: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["tracking"]["tracker"] = tracker_name

    temporary_config = Path(
        "configs/.comparison_config.yaml"
    )

    with open(
        temporary_config,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    start_time = time.perf_counter()

    result = subprocess.run(
        [
            "python",
            "-m",
            "scripts.run_tracking",
            "--config",
            str(temporary_config),
        ],
        check=True,
    )

    elapsed = time.perf_counter() - start_time

    temporary_config.unlink(
        missing_ok=True
    )

    return result.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Compare ByteTrack and BoT-SORT."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    args = parser.parse_args()

    trackers = [
        "bytetrack.yaml",
        "botsort.yaml",
    ]

    print(
        "\n========== TRACKER COMPARISON ==========\n"
    )

    for tracker_name in trackers:
        print(
            f"\nRunning tracker: {tracker_name}"
        )

        _, elapsed = run_tracker(
            args.config,
            tracker_name,
        )

        print(
            f"Total wall-clock time: "
            f"{elapsed:.2f} seconds"
        )

    print(
        "\n========================================\n"
    )


if __name__ == "__main__":
    main()
