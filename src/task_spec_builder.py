import json
from dataclasses import asdict
from pathlib import Path

from src.schemas import TaskSpec

def build_pxr_activity_task_spec(data_dir: str) -> TaskSpec:
    return TaskSpec(
        challenge_name="openadmet/pxr-challenge",
        track="activity",
        task_type="small_molecule_regression",
        target_column="pEC50",
        primary_metric="RAE",
        submission_columns=["SMILES", "Molecule Name", "pEC50"],
        data_dir=data_dir,
    )


def write_task_spec(task_spec: TaskSpec, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(asdict(task_spec), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )