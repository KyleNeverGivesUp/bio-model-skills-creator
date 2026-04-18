import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.constants import RUN_ID
from src.data_utils import (
    load_activity_test,
    load_activity_train,
    validate_activity_dataset,
    write_dataset_report,
)
from src.exporter import export_best_submission
from src.planner import build_mvp_plans, write_plans
from src.runner import run_plan
from src.schemas import RunState
from src.selector import load_all_metrics, select_best_plan, write_best_plan, write_leaderboard
from src.task_spec_builder import build_pxr_activity_task_spec, write_task_spec
from src.tuner import run_tuner


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def write_run_state(run_state: RunState, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(asdict(run_state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def update_run_state(run_state: RunState, run_dir: Path) -> None:
    write_run_state(run_state, run_dir / "run_state.json")


def run_selector_stage(run_dir: Path, primary_metric: str) -> dict:
    metrics_list = load_all_metrics(str(run_dir))
    best_plan = select_best_plan(metrics_list, primary_metric=primary_metric)
    write_leaderboard(metrics_list, run_dir / "leaderboard.json", primary_metric=primary_metric)
    write_best_plan(best_plan, run_dir / "best_plan.json")
    return best_plan


def run_baseline_stage(run_id: str, data_dir: str, run_dir: Path, primary_metric: str) -> list[str]:
    logger.info("Loading train/test data from %s", data_dir)
    train_df = load_activity_train(data_dir)
    test_df = load_activity_test(data_dir)

    task_spec = build_pxr_activity_task_spec(data_dir)
    plans = build_mvp_plans(task_spec, run_id)

    supported_plans = []
    for plan in plans:
        if plan.feature_type == "morgan_fingerprint" and plan.model_type in {"ridge", "random_forest"}:
            supported_plans.append(plan)
        else:
            logger.info(
                "Skipping unsupported baseline plan_id=%s feature_type=%s model_type=%s",
                plan.plan_id,
                plan.feature_type,
                plan.model_type,
            )

    result_paths: list[str] = []
    for plan in supported_plans:
        output_dir = run_dir / "plans" / plan.plan_id
        logger.info(
            "Running baseline plan_id=%s feature_type=%s model_type=%s",
            plan.plan_id,
            plan.feature_type,
            plan.model_type,
        )
        metrics = run_plan(plan, train_df, test_df, str(output_dir))
        logger.info(
            "Finished baseline plan_id=%s mae=%.6f r2=%.6f",
            plan.plan_id,
            metrics["mae"],
            metrics["r2"],
        )
        result_paths.append(str(output_dir / "metrics.json"))

    best_baseline = run_selector_stage(run_dir, primary_metric=primary_metric)
    logger.info(
        "Selected best baseline plan_id=%s mae=%.6f r2=%.6f",
        best_baseline["plan_id"],
        best_baseline["mae"],
        best_baseline["r2"],
    )
    return result_paths


def run_pxr_activity_mvp(
    run_id: str = RUN_ID,
    data_dir: str = "data/pxr_activity",
    primary_metric: str = "RAE",
) -> dict:
    run_dir = Path("outputs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting manager pipeline for run_id=%s", run_id)

    run_state = RunState(run_id=run_id, current_stage="init")
    update_run_state(run_state, run_dir)

    run_state.current_stage = "validate_data"
    update_run_state(run_state, run_dir)
    logger.info("Validating dataset in %s", data_dir)
    dataset_report = validate_activity_dataset(data_dir)
    write_dataset_report(dataset_report, run_dir / "dataset_report.json")
    if not dataset_report["valid"]:
        raise ValueError(f"Dataset validation failed: {dataset_report['errors']}")

    run_state.current_stage = "build_task_spec"
    update_run_state(run_state, run_dir)
    logger.info("Building task spec")
    task_spec = build_pxr_activity_task_spec(data_dir)
    task_spec_path = run_dir / "task_spec.json"
    write_task_spec(task_spec, task_spec_path)
    run_state.task_spec_path = str(task_spec_path)
    update_run_state(run_state, run_dir)

    run_state.current_stage = "build_plans"
    update_run_state(run_state, run_dir)
    logger.info("Building design plans")
    plans = build_mvp_plans(task_spec, run_id)
    design_plans_path = run_dir / "design_plans.json"
    write_plans(plans, design_plans_path)
    run_state.plan_paths = [str(design_plans_path)]
    update_run_state(run_state, run_dir)

    run_state.current_stage = "run_baselines"
    update_run_state(run_state, run_dir)
    run_state.result_paths = run_baseline_stage(run_id, data_dir, run_dir, primary_metric)
    run_state.best_plan_path = str(run_dir / "best_plan.json")
    update_run_state(run_state, run_dir)

    run_state.current_stage = "tune_best"
    update_run_state(run_state, run_dir)
    logger.info("Running tuner on current best baseline")
    best_tuned = run_tuner(run_id=run_id, data_dir=data_dir, primary_metric=primary_metric)
    logger.info(
        "Best tuned trial plan_id=%s mae=%.6f r2=%.6f",
        best_tuned["plan_id"],
        best_tuned["mae"],
        best_tuned["r2"],
    )

    run_state.current_stage = "select_best_overall"
    update_run_state(run_state, run_dir)
    logger.info("Selecting best overall plan from baseline + tuned trials")
    best_overall = run_selector_stage(run_dir, primary_metric=primary_metric)
    run_state.best_plan_path = str(run_dir / "best_plan.json")
    update_run_state(run_state, run_dir)
    logger.info(
        "Selected best overall plan_id=%s mae=%.6f r2=%.6f",
        best_overall["plan_id"],
        best_overall["mae"],
        best_overall["r2"],
    )

    run_state.current_stage = "export_submission"
    update_run_state(run_state, run_dir)
    logger.info("Exporting final submission")
    final_report = export_best_submission(run_id=run_id, data_dir=data_dir)

    run_state.current_stage = "completed"
    update_run_state(run_state, run_dir)
    logger.info("Pipeline completed successfully")
    logger.info("Submission path: %s", final_report["submission_path"])
    logger.info("Best plan id: %s", final_report["best_plan"]["plan_id"])

    return {
        "run_id": run_id,
        "submission_path": final_report["submission_path"],
        "best_plan": final_report["best_plan"],
        "run_state_path": str(run_dir / "run_state.json"),
    }


if __name__ == "__main__":
    result = run_pxr_activity_mvp()
    print("pxr manager completed:")
    print(f"  run_id: {result['run_id']}")
    print(f"  submission_path: {result['submission_path']}")
    print(f"  best_plan_id: {result['best_plan']['plan_id']}")
    print(f"  run_state_path: {result['run_state_path']}")
