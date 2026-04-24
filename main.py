import argparse

from src.train import train_and_compare_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake news detection project entry")
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "compare"],
        help="Execution mode",
    )
    parser.add_argument(
        "--data-source",
        default="local",
        choices=["local", "kagglehub"],
        help="Data source for training",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Local data directory (also used as kagglehub cache)",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="clmentbisaillon/fake-and-real-news-dataset",
        help="Kaggle dataset id when --data-source kagglehub",
    )
    args = parser.parse_args()

    if args.mode in {"train", "compare"}:
        result = train_and_compare_models(
            data_source=args.data_source,
            data_dir=args.data_dir,
            kaggle_dataset=args.kaggle_dataset,
        )
        print("Training and comparison finished")
        print(result["metrics"])  # noqa: T201
        print(f"Saved comparison: {result['comparison_path']}")  # noqa: T201


if __name__ == "__main__":
    main()
