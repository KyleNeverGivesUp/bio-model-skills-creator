import json
from dataclasses import asdict
from pathlib import Path

from src.schemas import PlanSpec, TaskSpec


def build_mvp_plans(task_spec: TaskSpec, run_id: str) -> list[PlanSpec]:

    plans = [
        PlanSpec(
            plan_id="morgan_ridge",
            name="Morgan Fingerprint + Ridge",
            feature_type="morgan_fingerprint",
            model_type="ridge",
            params={
                "radius": 2,
                "n_bits": 2048,
                "alpha": 1.0,
            },
            notes="Low-variance linear baseline for small-molecule regression.",
        ),
        PlanSpec(
            plan_id="morgan_rf",
            name="Morgan Fingerprint + RandomForest",
            feature_type="morgan_fingerprint",
            model_type="random_forest",
            params={
                "radius": 2,
                "n_bits": 2048,
                "n_estimators": 300,
                "max_depth": 6,
                "random_state": 42,
            },
            notes="Tree-based non-linear baseline on Morgan fingerprints.",
        ),
        PlanSpec(
            plan_id="rdkit_rf",
            name="RDKit Descriptors + RandomForest",
            feature_type="rdkit_descriptors",
            model_type="random_forest",
            params={
                "n_estimators": 300,
                "max_depth": 12,
                "random_state": 42,
            },
            notes="Descriptor-based baseline using tabular molecular features.",
        ),
    ]

    return plans


def write_plans(plans: list[PlanSpec], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plan_dicts = [asdict(plan) for plan in plans]

    output_file.write_text(
        json.dumps({"plans": plan_dicts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
