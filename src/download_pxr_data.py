from pathlib import Path

import pandas as pd


TRAIN_URL = "hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv"
TEST_URL = "hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv"


def download_pxr_activity_data(output_dir: str = "data/pxr_activity") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_URL)
    test_df = pd.read_csv(TEST_URL)

    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    sample_submission = test_df[["SMILES", "Molecule Name"]].copy()
    sample_submission["pEC50"] = 0.0
    sample_submission.to_csv(output_path / "sample_submission.csv", index=False)


if __name__ == "__main__":
    download_pxr_activity_data()
