from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json

@dataclass
class TaskSpec:
    challenge_name: str
    track: str
    task_type: str
    target_column: str
    primary_metric: str
    submission_columns: list[str]
    data_dir: str


@dataclass
class PlanSpec:
    plan_id: str
    name: str
    feature_type: str
    model_type: str
    params: dict[str, Any] = field(default_factory=dict) # model params
    notes: str = ""


@dataclass
class RunState:
    run_id: str
    task_spec_path: str | None = None
    current_stage: str = "init"
    plan_paths: list[str] = field(default_factory=list)
    result_paths: list[str] = field(default_factory=list)
    best_plan_path: str | None = None

def to_dict(obj):
    return asdict(obj)

def write_json(data: dict, output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )