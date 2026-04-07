"""
Baseline Runner for Curator Environment
=========================================

Runs three baseline policies against the Curator environment and reports scores.

Policies:
    random      — Recommend k random items (no filtering or ranking).
    heuristic   — Filter low-community-score items, rank by score, recommend top-k.
    llm         — Use an LLM via OpenAI-compatible API (same logic as inference.py).

Environment Variables (LLM policy only):
    API_BASE_URL   The API endpoint for the LLM (default: HuggingFace router).
    MODEL_NAME     The model identifier (default: meta-llama/Llama-3.2-1B-Instruct).
    HF_TOKEN       API key (also reads API_KEY).

Usage:
    # Start the server first:
    #   uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Run all policies on all tasks:
    python run_baselines.py

    # Run a specific policy and task:
    python run_baselines.py --policy heuristic --task easy

    # Run with LLM policy:
    HF_TOKEN=<token> python run_baselines.py --policy llm

    # Point to a Docker container or remote server:
    python run_baselines.py --url ws://localhost:8001
"""

import argparse
import asyncio
import json
import os
import random as rand_module
import textwrap
from typing import Any, Dict, List, Optional

from client import CuratorEnv
from models import CuratorAction

DEFAULT_URL = "ws://localhost:8000"
SEEDS = [42, 123, 456]
TASKS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# LLM helpers (reused from inference.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a content curation agent. You help users find the most relevant
    articles from a pool of content items based on their interest profile.

    Available actions (respond with valid JSON):

    1. Filter (remove irrelevant items):
       {"action_type": "filter", "item_ids": ["id1", "id2", ...]}

    2. Categorize items:
       {"action_type": "categorize", "categories": {"id1": "urgent", "id2": "skip", ...}}
       Categories: "urgent", "read_later", "share", "skip"

    3. Rank items by relevance:
       {"action_type": "rank", "rankings": ["best_id", "second_id", ...]}

    4. Final recommendation (ends episode):
       {"action_type": "recommend", "item_ids": ["id1", "id2", ...]}

    Strategy: First filter out clearly irrelevant items, then rank the remainder,
    then recommend the top items.

    IMPORTANT: Respond with ONLY a JSON object, no markdown or explanation.
""").strip()


def parse_action_from_response(text: str) -> Optional[Dict]:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def format_items_for_prompt(items: List[Dict], max_items: int = 30) -> str:
    lines = []
    for item in items[:max_items]:
        tags = ", ".join(item.get("tags", []))
        lines.append(
            f"- [{item['id']}] ({item['source']}) {item['title']} [tags: {tags}] [score: {item.get('score', 0)}]"
        )
    if len(items) > max_items:
        lines.append(f"  ... and {len(items) - max_items} more items")
    return "\n".join(lines)


def format_profile_for_prompt(profile: Dict) -> str:
    interests = ", ".join(
        f"{k}={v:.1f}"
        for k, v in sorted(profile.get("interests", {}).items(), key=lambda x: -x[1])
    )
    sources = ", ".join(profile.get("preferred_sources", [])) or "no preference"
    history = profile.get("read_history", [])
    return (
        f"Interests: {interests}\n"
        f"Preferred sources: {sources}\n"
        f"Skill level: {profile.get('skill_level', 'intermediate')}\n"
        f"Time budget: {profile.get('time_budget_mins', 60)} mins\n"
        f"Already read: {len(history)} items ({', '.join(history[:5])}{'...' if len(history) > 5 else ''})"
    )


def build_user_prompt(obs: Any, step: int, last_feedback: Optional[str]) -> str:
    items = [
        item.model_dump() if hasattr(item, "model_dump") else item for item in obs.items
    ]
    profile = (
        obs.user_profile.model_dump()
        if hasattr(obs.user_profile, "model_dump")
        else obs.user_profile
    )
    ti = obs.task_info
    prompt = f"""Step {step}/{ti.max_steps}. You must recommend {ti.recommend_k} items.
Pool: {ti.pool_size} items. Filtered so far: {ti.items_filtered}. Categorized: {ti.items_categorized}.

User Profile:
{format_profile_for_prompt(profile)}

Items in pool:
{format_items_for_prompt(items)}
"""
    if last_feedback:
        prompt += f"\nLast action feedback: {last_feedback}\n"
    if step >= ti.max_steps - 1:
        prompt += (
            "\nWARNING: This is your last step. You MUST use 'recommend' action now.\n"
        )
    elif step >= ti.max_steps - 2:
        prompt += "\nOnly 2 steps left. Consider recommending soon.\n"
    return prompt


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


async def run_random(env: CuratorEnv, task_id: str, seed: int) -> float:
    """Recommend k random items with no processing."""
    result = await env.reset(task_id=task_id, seed=seed)
    obs = result.observation
    k = obs.task_info.recommend_k

    rand_module.seed(seed)
    ids = [it.id for it in obs.items]
    rand_module.shuffle(ids)

    action = CuratorAction(action_type="recommend", item_ids=ids[:k])
    result = await env.step(action)
    return result.reward or 0.0


async def run_heuristic(env: CuratorEnv, task_id: str, seed: int) -> float:
    """Filter low-community-score items, rank by score, recommend top-k."""
    result = await env.reset(task_id=task_id, seed=seed)
    obs = result.observation
    k = obs.task_info.recommend_k

    # Filter bottom half by community score
    sorted_items = sorted(obs.items, key=lambda x: x.score, reverse=True)
    keep_count = max(k + 5, len(obs.items) // 2)
    filter_ids = [it.id for it in sorted_items[keep_count:]]

    if filter_ids:
        action = CuratorAction(action_type="filter", item_ids=filter_ids)
        result = await env.step(action)
        obs = result.observation

    # Rank by score
    ranked_ids = [
        it.id for it in sorted(obs.items, key=lambda x: x.score, reverse=True)
    ]
    action = CuratorAction(action_type="rank", rankings=ranked_ids)
    result = await env.step(action)

    # Recommend top-k
    action = CuratorAction(action_type="recommend", item_ids=ranked_ids[:k])
    result = await env.step(action)
    return result.reward or 0.0


async def run_llm(env: CuratorEnv, task_id: str, seed: int) -> float:
    """Use an LLM to drive the agent (same logic as inference.py)."""
    from openai import OpenAI

    api_base = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.2-1B-Instruct"
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    client = OpenAI(base_url=api_base, api_key=api_key)

    result = await env.reset(task_id=task_id, seed=seed)
    obs = result.observation
    ti = obs.task_info
    max_steps = ti.max_steps if ti else 10
    k = ti.recommend_k if ti else 5
    last_feedback: Optional[str] = None
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(1, max_steps + 1):
        if result.done:
            break

        user_prompt = build_user_prompt(obs, step, last_feedback)
        messages.append({"role": "user", "content": user_prompt})

        action_dict = None
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            messages.append({"role": "assistant", "content": text})
            action_dict = parse_action_from_response(text)
            if action_dict and "action_type" not in action_dict:
                action_dict = None
        except Exception as exc:
            print(f"  [DEBUG] Model request failed: {exc}", flush=True)

        if action_dict is None:
            # Fallback: recommend first k items
            ids = [it.id for it in obs.items]
            action_dict = {"action_type": "recommend", "item_ids": ids[:k]}

        action = CuratorAction(**action_dict)
        result = await env.step(action)
        obs = result.observation

        if obs.feedback:
            last_feedback = obs.feedback.explanation
        else:
            last_feedback = None

        if result.done:
            break

    return result.reward or 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

POLICIES = {
    "random": run_random,
    "heuristic": run_heuristic,
    "llm": run_llm,
}


async def run_policy(
    policy_name: str, url: str, tasks: List[str], seeds: List[int]
) -> Dict:
    policy_fn = POLICIES[policy_name]
    results: Dict[str, List[float]] = {}

    for task_id in tasks:
        scores = []
        for seed in seeds:
            async with CuratorEnv(base_url=url) as env:
                score = await policy_fn(env, task_id, seed)
                scores.append(score)
        results[task_id] = scores

    return results


def print_results(policy_name: str, results: Dict[str, List[float]]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Policy: {policy_name}")
    print(f"{'=' * 60}")
    print(f"  {'Task':<10} {'Avg':>8} {'Min':>8} {'Max':>8}  Seeds")
    print(f"  {'-' * 54}")
    for task_id, scores in results.items():
        avg = sum(scores) / len(scores)
        mn, mx = min(scores), max(scores)
        print(
            f"  {task_id:<10} {avg:>8.3f} {mn:>8.3f} {mx:>8.3f}  {[round(s, 3) for s in scores]}"
        )


async def main():
    parser = argparse.ArgumentParser(
        description="Run baseline policies for Curator Environment"
    )
    parser.add_argument(
        "--policy",
        choices=list(POLICIES.keys()),
        default=None,
        help="Policy to run (default: all non-LLM policies)",
    )
    parser.add_argument(
        "--task", choices=TASKS, default=None, help="Task to run (default: all)"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"WebSocket URL of the server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help=f"Seeds for reproducibility (default: {SEEDS})",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS

    if args.policy:
        policies = [args.policy]
    else:
        # Default: run random + heuristic. LLM requires explicit opt-in.
        policies = ["random", "heuristic"]

    print(f"Server:   {args.url}")
    print(f"Tasks:    {tasks}")
    print(f"Seeds:    {args.seeds}")
    print(f"Policies: {policies}")

    for policy_name in policies:
        results = await run_policy(policy_name, args.url, tasks, args.seeds)
        print_results(policy_name, results)

    print()


if __name__ == "__main__":
    asyncio.run(main())
