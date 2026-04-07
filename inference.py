"""
Inference Script for Curator Environment
============================================

Environment Variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     Docker image name for the environment.
    CURATOR_TASK  Task difficulty: "easy", "medium", or "hard" (default: "easy").

STDOUT FORMAT:
    [START] task=<task_name> env=curator model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import CuratorEnv
from models import CuratorAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.2-1B-Instruct"
TASK_NAME = os.getenv("CURATOR_TASK", "hard")
BENCHMARK = "curator"
TEMPERATURE = 0.3
MAX_TOKENS = 2000
SUCCESS_SCORE_THRESHOLD = 0.3

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


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def format_items_for_prompt(items: List[Dict], max_items: int = 30) -> str:
    """Format content items into a compact string for the LLM prompt."""
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
    """Format user profile for the LLM prompt."""
    interests = ", ".join(f"{k}={v:.1f}" for k, v in sorted(profile.get("interests", {}).items(), key=lambda x: -x[1]))
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
    """Build the user prompt from current observation."""
    items = [item.model_dump() if hasattr(item, "model_dump") else item for item in obs.items]
    profile = obs.user_profile.model_dump() if hasattr(obs.user_profile, "model_dump") else obs.user_profile

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
        prompt += "\nWARNING: This is your last step. You MUST use 'recommend' action now.\n"
    elif step >= ti.max_steps - 2:
        prompt += "\nOnly 2 steps left. Consider recommending soon.\n"

    return prompt


def parse_action_from_response(text: str) -> Optional[Dict]:
    """Parse a JSON action from LLM response text."""
    text = text.strip()

    # Try to extract JSON from markdown code blocks
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

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


def get_model_action(
    client: OpenAI,
    obs: Any,
    step: int,
    last_feedback: Optional[str],
    messages: List[Dict[str, str]],
) -> Dict:
    """Get action from LLM, maintaining conversation history."""
    user_prompt = build_user_prompt(obs, step, last_feedback)
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": text})
        action = parse_action_from_response(text)
        if action and "action_type" in action:
            return action
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)

    # Fallback: recommend first N items from pool
    item_ids = [item.id if hasattr(item, "id") else item["id"] for item in obs.items]
    k = obs.task_info.recommend_k if obs.task_info else 5
    return {"action_type": "recommend", "item_ids": item_ids[:k]}


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with CuratorEnv(base_url="http://localhost:8000") as env:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        last_feedback: Optional[str] = None

        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

        result = await env.reset(task_id=TASK_NAME)
        obs = result.observation

        task_info = obs.task_info
        max_steps = task_info.max_steps if task_info else 10
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_dict = get_model_action(llm_client, obs, step, last_feedback, messages)
            action = CuratorAction(**action_dict)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            # Summarize action for logging
            if action.action_type == "categorize" and action.categories:
                action_summary = f"categorize({len(action.categories)}items)"
            elif action.action_type == "rank" and action.rankings:
                action_summary = f"rank({len(action.rankings)}items)"
            else:
                action_summary = f"{action.action_type}({len(action.item_ids)}items)"
            log_step(step=step, action=action_summary, reward=reward, done=done, error=error)

            # Capture feedback for next prompt
            if obs.feedback:
                last_feedback = obs.feedback.explanation
            else:
                last_feedback = None

            if done:
                break

        # Final score is the last reward (from recommend action)
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
