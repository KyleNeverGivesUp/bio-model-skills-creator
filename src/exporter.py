import json
from pathlib import Path

import pandas as pd

from src.constants import RUN_ID


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: dict, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_submission(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    required_pred_cols = {"SMILES", "Molecule Name", "pEC50"}
    missing_pred_cols = required_pred_cols - set(pred_df.columns)
    if missing_pred_cols:
        raise ValueError(f"Prediction file missing columns: {sorted(missing_pred_cols)}")

    submission = pred_df[["SMILES", "Molecule Name", "pEC50"]].copy()

    if len(submission) != len(test_df):
        raise ValueError(
            f"Submission row count {len(submission)} does not match test row count {len(test_df)}"
        )

    if submission["SMILES"].isna().any() or (submission["SMILES"].astype(str).str.strip() == "").any():
        raise ValueError("Submission contains empty SMILES values.")

    if submission["Molecule Name"].isna().any() or (
        submission["Molecule Name"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("Submission contains empty Molecule Name values.")

    if submission["pEC50"].isna().any():
        raise ValueError("Submission contains empty pEC50 values.")

    return submission


def write_submission(submission_df: pd.DataFrame, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_file, index=False)


def write_final_report(report: dict, output_path: str | Path) -> None:
    write_json(report, output_path)


def export_best_submission(run_id: str, data_dir: str = "data/pxr_activity") -> dict:
    run_dir = Path("outputs") / run_id
    best_plan_path = run_dir / "best_plan.json"
    best_plan = load_json(best_plan_path)

    if "metrics_path" in best_plan:
        predictions_path = Path(best_plan["metrics_path"]).parent / "test_predictions.csv"
    else:
        predictions_path = run_dir / "plans" / best_plan["plan_id"] / "test_predictions.csv"

    test_path = Path(data_dir) / "test.csv"
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(predictions_path)

    submission_df = build_submission(test_df, pred_df)

    submission_path = run_dir / "submission.csv"
    final_report_path = run_dir / "final_report.json"

    write_submission(submission_df, submission_path)

    final_report = {
        "run_id": run_id,
        "challenge_name": "openadmet/pxr-challenge",
        "track": "activity",
        "submission_path": str(submission_path),
        "best_plan": best_plan,
        "test_predictions_path": str(predictions_path),
        "n_submission_rows": int(len(submission_df)),
        "submission_columns": list(submission_df.columns),
    }
    write_final_report(final_report, final_report_path)

    return final_report


if __name__ == "__main__":
    report = export_best_submission(run_id=RUN_ID)
    print("submission exported:")
    print(f"  run_id: {report['run_id']}")
    print(f"  submission_path: {report['submission_path']}")
    print(f"  best_plan_id: {report['best_plan']['plan_id']}")
