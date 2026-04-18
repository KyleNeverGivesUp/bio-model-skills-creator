import json

from src.agent.base import LLMJsonAgent
from src.agent.types import AgentResult, RunContext


class RetrievalAgent(LLMJsonAgent):
    name = "retrieval"

    def run(self, context: RunContext) -> AgentResult:
        task_spec = json.loads((context.run_dir / "task_spec.json").read_text(encoding="utf-8"))
        llm_log_path = context.run_dir / "llm_logs" / "retrieval.json"

        system_prompt = (
            "You are a RetrievalAgent for a biological ML AutoML system. "
            "Your job is to decide whether to use an existing model-skill route or chemistry baselines. "
            "Return JSON only."
        )
        user_prompt = f"""
Task spec:
{json.dumps(task_spec, indent=2)}

For the current implementation, the only executable retrieval target is:
- chemistry_baselines

Respond with JSON exactly in this shape:
{{
  "mode": "fallback or retrieved",
  "selected_strategy": "chemistry_baselines",
  "reason": "<short reason>"
}}
"""
        used_fallback = False
        try:
            decision = self.call_json_logged(context, "retrieval", system_prompt, user_prompt)
            if decision.get("selected_strategy") != "chemistry_baselines":
                raise ValueError("Unsupported selected_strategy from LLM.")
        except Exception as exc:
            used_fallback = True
            decision = {
                "mode": "fallback",
                "selected_strategy": "chemistry_baselines",
                "reason": "Fallback to supported chemistry baselines.",
                "fallback_error": str(exc),
            }

        output_path = context.run_dir / "retrieval_result.json"
        output_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary="Produced retrieval decision.",
            outputs={
                "retrieval_result_path": str(output_path),
                "decision": decision,
                "used_fallback": used_fallback,
                "llm_log_path": str(llm_log_path),
            },
        )
