import json
from dataclasses import asdict

from src.agent.base import LLMJsonAgent
from src.agent.types import AgentResult, RunContext
from src.data_utils import load_activity_test, load_activity_train
from src.schemas import PlanSpec
from src.tuner import (
    build_tuning_trials,
    find_base_plan,
    load_best_plan,
    load_design_plans,
    run_tuning_trials,
    write_tuning_summary,
    write_tuning_trials,
)


class TunerAgent(LLMJsonAgent):
    name = "tuner"

    def _validate_trials(self, payload: dict) -> list[PlanSpec]:
        return [PlanSpec(**trial) for trial in payload.get("trials", [])]

    def run(self, context: RunContext) -> AgentResult:
        best_baseline = load_best_plan(context.run_dir / "best_plan.json")
        design_plans = load_design_plans(context.run_dir / "design_plans.json")
        base_plan = find_base_plan(best_baseline, design_plans)
        llm_log_path = context.run_dir / "llm_logs" / "tuner.json"

        system_prompt = (
            "You are a TunerAgent for a biological ML AutoML system. "
            "Choose tuning trials for the provided base plan. Return JSON only."
        )
        user_prompt = f"""
Best baseline:
{json.dumps(best_baseline, indent=2)}

Base plan:
{json.dumps(asdict(base_plan), indent=2, ensure_ascii=False)}

To preserve parity with the current manual tuning flow, use the canonical tuning search spaces:
- If model_type is ridge:
  alpha in [0.1, 1.0, 10.0]
- If model_type is random_forest:
  1. n_estimators=100, max_depth=4
  2. n_estimators=300, max_depth=6
  3. n_estimators=500, max_depth=8

Return JSON exactly in this shape:
{{
  "summary": "<short summary>",
  "trials": [
    {{
      "plan_id": "<trial id>",
      "name": "<trial name>",
      "feature_type": "{base_plan.feature_type}",
      "model_type": "{base_plan.model_type}",
      "params": {{...}},
      "notes": "<short note>"
    }}
  ]
}}
"""

        used_fallback = False
        try:
            payload = self.call_json_logged(context, "tuner", system_prompt, user_prompt)
            trials = self._validate_trials(payload)
            if not trials:
                raise ValueError("No tuning trials returned by LLM.")
        except Exception as exc:
            used_fallback = True
            payload = {
                "summary": "Fallback to deterministic tuning grid.",
                "fallback_error": str(exc),
            }
            trials = build_tuning_trials(base_plan)

        write_tuning_trials(trials, context.run_dir / "tuning_trials.json")

        train_df = load_activity_train(context.data_dir)
        test_df = load_activity_test(context.data_dir)
        tuned_results = run_tuning_trials(trials, train_df, test_df, context.run_dir)

        write_tuning_summary(
            base_plan=base_plan,
            best_baseline=best_baseline,
            tuned_results=tuned_results,
            output_path=context.run_dir / "tuning_summary.json",
            primary_metric=context.primary_metric,
        )

        report_path = context.run_dir / "tuner_report.json"
        report_path.write_text(
            json.dumps(
                {
                    **payload,
                    "used_fallback": used_fallback,
                    "llm_log_path": str(llm_log_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary="Completed LLM-guided tuning trials.",
            outputs={
                "tuning_trials_path": str(context.run_dir / "tuning_trials.json"),
                "tuning_summary_path": str(context.run_dir / "tuning_summary.json"),
                "tuner_report_path": str(report_path),
                "used_fallback": used_fallback,
                "llm_log_path": str(llm_log_path),
            },
        )
