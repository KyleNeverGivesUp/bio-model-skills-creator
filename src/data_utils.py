import json
from pathlib import Path

import pandas as pd

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

REQUIRED_TRAIN_COLUMNS = ["SMILES", "Molecule Name", "pEC50"]
REQUIRED_TEST_COLUMNS = ["SMILES", "Molecule Name"]

def load_activity_train(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir) / TRAIN_FILE_NAME
    return pd.read_csv(data_path)

def load_activity_test(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir) / TEST_FILE_NAME
    return pd.read_csv(data_path)

def _check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [col for col in required_columns if col not in df.columns]

def _count_empty_values(df: pd.DataFrame, column_name: str) -> int:
    if column_name not in df.columns:
        return -1
    return int(df[column_name].isna().sum() + (df[column_name].astype(str).str.strip() == "").sum())

def validate_activity_dataset(data_dir: str) -> dict:
    data_path = Path(data_dir)
    train_path = data_path / TRAIN_FILE_NAME
    test_path = data_path / TEST_FILE_NAME

    report = {
        "data_dir": str(data_path),
        "files": {
            "train_exists": train_path.exists(),
            "test_exists": test_path.exists(),
            "train_path": str(train_path),
            "test_path": str(test_path),
        },
        "train": {
            "readable": False,
            "row_count": 0,
            "missing_required_columns": [],
            "empty_smiles_count": -1,
            "empty_molecule_name_count": -1,
        },
        "test": {
            "readable": False,
            "row_count": 0,
            "missing_required_columns": [],
            "empty_smiles_count": -1,
            "empty_molecule_name_count": -1,
        },
        "valid": False,
        "errors": [],
    }

    if not train_path.exists():
        report["errors"].append(f"Missing train file: {train_path}")

    if not test_path.exists():
        report["errors"].append(f"Missing test file: {test_path}")

    train_df = None
    test_df = None

    if train_path.exists():
        try:
            train_df = pd.read_csv(train_path)
            report["train"]["readable"] = True
            report["train"]["row_count"] = int(len(train_df))
            report["train"]["missing_required_columns"] = _check_required_columns(
                train_df, REQUIRED_TRAIN_COLUMNS
            )
            report["train"]["empty_smiles_count"] = _count_empty_values(train_df, "SMILES")
            report["train"]["empty_molecule_name_count"] = _count_empty_values(train_df, "Molecule Name")
        except Exception as exc:
            report["errors"].append(f"Could not read train file: {exc}")

    if test_path.exists():
        try:
            test_df = pd.read_csv(test_path)
            report["test"]["readable"] = True
            report["test"]["row_count"] = int(len(test_df))
            report["test"]["missing_required_columns"] = _check_required_columns(
                test_df, REQUIRED_TEST_COLUMNS
            )
            report["test"]["empty_smiles_count"] = _count_empty_values(test_df, "SMILES")
            report["test"]["empty_molecule_name_count"] = _count_empty_values(test_df, "Molecule Name")
        except Exception as exc:
            report["errors"].append(f"Could not read test file: {exc}")

    if report["train"]["readable"] and report["train"]["row_count"] <= 0:
        report["errors"].append("Train file has zero rows.")

    if report["test"]["readable"] and report["test"]["row_count"] <= 0:
        report["errors"].append("Test file has zero rows.")

    if report["train"]["missing_required_columns"]:
        report["errors"].append(
            f"Train file is missing required columns: {report['train']['missing_required_columns']}"
        )

    if report["test"]["missing_required_columns"]:
        report["errors"].append(
            f"Test file is missing required columns: {report['test']['missing_required_columns']}"
        )

    if report["train"]["empty_smiles_count"] > 0:
        report["errors"].append(
            f"Train file contains {report['train']['empty_smiles_count']} empty SMILES values."
        )

    if report["test"]["empty_smiles_count"] > 0:
        report["errors"].append(
            f"Test file contains {report['test']['empty_smiles_count']} empty SMILES values."
        )

    if report["train"]["empty_molecule_name_count"] > 0:
        report["errors"].append(
            f"Train file contains {report['train']['empty_molecule_name_count']} empty Molecule Name values."
        )

    if report["test"]["empty_molecule_name_count"] > 0:
        report["errors"].append(
            f"Test file contains {report['test']['empty_molecule_name_count']} empty Molecule Name values."
        )

    report["valid"] = len(report["errors"]) == 0
    return report


def write_dataset_report(report: dict, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
