import json
from dataclasses import asdict

from src.agent.base import LLMJsonAgent
from src.agent.types import AgentResult, RunContext
from src.planner import build_mvp_plans
from src.schemas import PlanSpec


class DesignerAgent(LLMJsonAgent):
    name = "designer"

    def _validate_plans(self, payload: dict) -> list[PlanSpec]:
        plans = payload.get("plans", [])
        validated: list[PlanSpec] = []
        for plan in plans:
            validated.append(PlanSpec(**plan))
        return validated

    def run(self, context: RunContext) -> AgentResult:
        task_spec = json.loads((context.run_dir / "task_spec.json").read_text(encoding="utf-8"))
        retrieval_result = json.loads((context.run_dir / "retrieval_result.json").read_text(encoding="utf-8"))
        llm_log_path = context.run_dir / "llm_logs" / "designer.json"

        system_prompt = (
            "You are a DesignerAgent for a biological ML AutoML system. "
            "Design executable plans only. Return JSON only."
        )
        user_prompt = f"""
Task spec:
{json.dumps(task_spec, indent=2)}

Retrieval result:
{json.dumps(retrieval_result, indent=2)}

Current execution layer supports these canonical baselines:
1. morgan_ridge:
   - feature_type: morgan_fingerprint
   - model_type: ridge
   - params: radius=2, n_bits=2048, alpha=1.0
2. morgan_rf:
   - feature_type: morgan_fingerprint
   - model_type: random_forest
   - params: radius=2, n_bits=2048, n_estimators=300, max_depth=6, random_state=42
3. rdkit_rf:
   - feature_type: rdkit_descriptors
   - model_type: random_forest
   - params: n_estimators=300, max_depth=12, random_state=42

For parity with the current manual pipeline, you must return exactly these three baseline plans.

Respond with JSON exactly in this shape:
{{
  "summary": "<short summary>",
  "plans": [
    {{
      "plan_id": "morgan_ridge",
      "name": "Morgan Fingerprint + Ridge",
      "feature_type": "morgan_fingerprint",
      "model_type": "ridge",
      "params": {{"radius": 2, "n_bits": 2048, "alpha": 1.0}},
      "notes": "<short note>"
    }},
    {{
      "plan_id": "morgan_rf",
      "name": "Morgan Fingerprint + RandomForest",
      "feature_type": "morgan_fingerprint",
      "model_type": "random_forest",
      "params": {{"radius": 2, "n_bits": 2048, "n_estimators": 300, "max_depth": 6, "random_state": 42}},
      "notes": "<short note>"
    }},
    {{
      "plan_id": "rdkit_rf",
      "name": "RDKit Descriptors + RandomForest",
      "feature_type": "rdkit_descriptors",
      "model_type": "random_forest",
      "params": {{"n_estimators": 300, "max_depth": 12, "random_state": 42}},
      "notes": "<short note>"
    }}
  ]
}}
"""

        used_fallback = False
        try:
            payload = self.call_json_logged(context, "designer", system_prompt, user_prompt)
            plans = self._validate_plans(payload)
        except Exception as exc:
            used_fallback = True
            plans = build_mvp_plans(
                task_spec=type("TaskSpecProxy", (), task_spec)(),
                run_id=context.run_id,
            )
            payload = {
                "summary": "Fallback to deterministic MVP planner.",
                "fallback_error": str(exc),
            }

        design_plans_path = context.run_dir / "design_plans.json"
        design_plans_path.write_text(
            json.dumps({"plans": [asdict(plan) for plan in plans]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        agent_report_path = context.run_dir / "designer_report.json"
        agent_report_path.write_text(
            json.dumps(
                {
                    "summary": payload.get("summary", ""),
                    "n_plans": len(plans),
                    "used_fallback": used_fallback,
                    "llm_log_path": str(llm_log_path),
                    "fallback_error": payload.get("fallback_error"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary=f"Generated {len(plans)} plans via LLM-guided planner.",
            outputs={
                "design_plans_path": str(design_plans_path),
                "designer_report_path": str(agent_report_path),
                "n_plans": len(plans),
                "used_fallback": used_fallback,
                "llm_log_path": str(llm_log_path),
            },
        )
