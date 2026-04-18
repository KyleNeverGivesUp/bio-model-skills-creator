"""
Manager Agent
Orchestrates a multi-agent ML pipeline using skills.

Flow:
1. Load name + description from all skills (metadata only)
2. Ask Claude to decide which skill to run next
3. Load the full SKILL.md for the chosen skill
4. Run the skill, update state
5. Loop until done
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
SKILLS_DIR = Path("./skills")


# ── Skill Loading ──────────────────────────────────────────────────────────

def parse_frontmatter(skill_path: Path) -> dict:
    """Parse only the YAML frontmatter from a SKILL.md file.
    Returns dict with 'name' and 'description'."""
    content = skill_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {"name": skill_path.parent.name, "description": ""}

    frontmatter = match.group(1)
    name = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
    desc = re.search(r"^description:\s*(.+?)(?=\n\w|\Z)", frontmatter, re.MULTILINE | re.DOTALL)
    
    return {
        "name": name.group(1).strip() if name else skill_path.parent.name,
        "description": desc.group(1).strip().replace("\n", " ") if desc else "",
    }


def load_all_metadata() -> dict:
    """Load name + description from all skills. Used by Manager to decide next step."""
    skills = {}
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        meta = parse_frontmatter(skill_md)
        skills[meta["name"]] = {
            "dir": str(skill_dir),
            "description": meta["description"],
        }
    return skills


def load_full_skill(skill_name: str, skills_meta: dict) -> str:
    """Load the full SKILL.md content for a chosen skill."""
    skill_info = skills_meta.get(skill_name)
    if not skill_info:
        raise ValueError(f"Skill '{skill_name}' not found.")
    skill_path = Path(skill_info["dir"]) / "SKILL.md"
    return skill_path.read_text(encoding="utf-8")


# ── Manager Decision ───────────────────────────────────────────────────────

def manager_decide(task: str, state: dict, skills_meta: dict) -> dict:
    """Ask Claude to decide which skill to run next, based on current state.
    Returns dict with 'next_skill' and 'instructions'."""
    
    # Build a compact summary of available skills
    skills_summary = "\n".join([
        f"- {name}: {info['description'][:150]}"
        for name, info in skills_meta.items()
    ])

    prompt = f"""You are a Manager Agent coordinating an ML pipeline.

Task: {task}

Current state:
{json.dumps(state, indent=2)}

Available skills:
{skills_summary}

Based on the current state, decide what to do next.
Respond ONLY with this JSON, no other text:
{{
  "next_skill": "<skill name, or 'stop' if pipeline is complete>",
  "instructions": "<specific instructions for the chosen skill>",
  "reason": "<one sentence explaining why>"
}}"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = resp.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Skill Execution ────────────────────────────────────────────────────────

def run_skill(skill_name: str, instructions: str, state: dict, skills_meta: dict) -> str:
    """Load full SKILL.md and execute the skill via Claude."""
    skill_content = load_full_skill(skill_name, skills_meta)

    prompt = f"""You are an AI agent. Follow the skill instructions below to complete your task.

{skill_content}

---
Current pipeline state:
{json.dumps(state, indent=2)}

Your specific instructions:
{instructions}

Complete the task. If you need to write code, write it clearly so it can be executed.
Return a JSON summary of what you did and the results:
{{
  "status": "done or failed",
  "summary": "<what was accomplished>",
  "outputs": {{<any key results, file paths, metrics>}}
}}"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()


def parse_skill_output(output: str) -> dict:
    """Try to extract JSON summary from skill output."""
    try:
        match = re.search(r"\{[\s\S]*\}", output)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"status": "done", "summary": output[:200], "outputs": {}}


# ── Main Loop ──────────────────────────────────────────────────────────────

def run_pipeline(task: str, max_steps: int = 10):
    print("=" * 60)
    print("Manager Agent")
    print(f"Task: {task}")
    print("=" * 60)

    # Load skill metadata once
    skills_meta = load_all_metadata()
    print(f"\nLoaded {len(skills_meta)} skills:")
    for name, info in skills_meta.items():
        print(f"  - {name}")

    # Initial state
    state = {
        "task": task,
        "steps_completed": [],
        "results": {}
    }

    for step in range(max_steps):
        print(f"\n{'─' * 40}")
        print(f"Step {step + 1}")

        # Manager decides next skill
        decision = manager_decide(task, state, skills_meta)
        next_skill = decision.get("next_skill", "stop")
        instructions = decision.get("instructions", "")
        reason = decision.get("reason", "")

        print(f"Manager → {next_skill}")
        print(f"Reason: {reason}")

        if next_skill == "stop":
            print("\nPipeline complete.")
            break

        if next_skill not in skills_meta:
            print(f"[WARN] Unknown skill '{next_skill}', stopping.")
            break

        # Run the chosen skill
        print(f"Running skill: {next_skill}...")
        output = run_skill(next_skill, instructions, state, skills_meta)
        result = parse_skill_output(output)

        # Update state
        state["steps_completed"].append(next_skill)
        state["results"][next_skill] = result
        print(f"Result: {result.get('summary', '')[:200]}")

    print("\n" + "=" * 60)
    print("Final State:")
    print(json.dumps(state, indent=2))
    print("=" * 60)
    return state


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task = """
    Kaggle competition: Retroviral Wall Challenge
    Goal: Predict which reverse transcriptases are active for prime editing.
    Metric: CLS (harmonic mean of PR-AUC and Weighted Spearman)
    Data directory: ./data/
    Files: train.csv (sequences + 66 biophysical features + labels), test.csv, esm2_embeddings.npz, family_splits.csv
    Required output: submission.csv with rt_name and predicted_score columns
    Use LOFO (Leave-One-Family-Out) cross-validation across 7 families.
    """
    run_pipeline(task)