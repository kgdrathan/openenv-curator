"""
Curator Environment Implementation.

A personalized content curation environment where an agent must filter,
categorize, rank, and recommend content items from a mixed pool of real
articles across multiple sources.
"""

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionFeedback,
        ContentItem,
        CuratorAction,
        CuratorObservation,
        TaskInfo,
        UserProfile,
    )
except ImportError:
    from models import (
        ActionFeedback,
        ContentItem,
        CuratorAction,
        CuratorObservation,
        TaskInfo,
        UserProfile,
    )

try:
    from . import grader
except ImportError:
    from server import grader

DATA_DIR = Path(__file__).parent.parent / "data"


class CuratorEnvironment(Environment):
    """
    Personalized content curation environment.

    The agent receives a pool of real content items and a user profile,
    then must filter, categorize, rank, and recommend the most relevant
    items. Scored using standard IR metrics (NDCG, precision, recall).

    Tasks:
        - easy: 20 items from 1 source, clear preferences
        - medium: 50 items from 3 sources, nuanced preferences
        - hard: 100 items from 4 sources, minimal initial preferences
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Load static data
        self._all_items = self._load_json("items.json")
        self._all_tasks = {t["task_id"]: t for t in self._load_json("tasks.json")}
        self._ground_truth = self._load_json("ground_truth.json")

        # Episode state
        self._task_config: Optional[dict] = None
        self._profile: Optional[dict] = None
        self._relevance: Dict[str, float] = {}
        self._current_pool: List[dict] = []
        self._items_by_id: Dict[str, dict] = {}
        self._filtered_ids: List[str] = []
        self._categories: Dict[str, str] = {}
        self._last_ranking: List[str] = []
        self._recommended_ids: List[str] = []
        self._items_filtered_count = 0
        self._items_categorized_count = 0

    @staticmethod
    def _load_json(filename: str) -> dict | list:
        path = DATA_DIR / filename
        with open(path) as f:
            return json.load(f)

    def reset(self, **kwargs) -> CuratorObservation:  # type: ignore[override]
        """Reset the environment with a task configuration.

        Args:
            **kwargs: Must include 'task_id' ("easy", "medium", or "hard").
                      Optional 'seed' for reproducibility.
        """
        task_id = kwargs.get("task_id", "easy")
        seed = kwargs.get("seed", None)

        if task_id not in self._all_tasks:
            task_id = "easy"

        self._task_config = self._all_tasks[task_id]
        self._profile = copy.deepcopy(self._task_config["profile"])
        self._relevance = self._ground_truth.get(task_id, {})

        # Select items for this task
        sources = self._task_config["sources"]
        item_count = self._task_config["item_count"]
        if sources == "all":
            pool = list(self._all_items)
        else:
            pool = [it for it in self._all_items if it["source"] in sources]

        # Shuffle with seed for reproducibility
        if seed is not None:
            random.seed(seed)
        random.shuffle(pool)
        self._current_pool = pool[:item_count]
        self._items_by_id = {it["id"]: it for it in self._current_pool}

        # Reset episode state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._filtered_ids = []
        self._categories = {}
        self._last_ranking = []
        self._recommended_ids = []
        self._items_filtered_count = 0
        self._items_categorized_count = 0

        return self._make_observation(reward=0.0, done=False)

    def step(self, action: CuratorAction) -> CuratorObservation:  # type: ignore[override]
        """Execute one step in the environment.

        Args:
            action: CuratorAction with action_type and relevant fields.
        """
        self._state.step_count += 1
        max_steps = self._task_config["max_steps"]

        action_type = action.action_type
        reward = 0.0
        feedback = ActionFeedback()
        done = False

        if action_type == "filter":
            reward, feedback = self._handle_filter(action)
        elif action_type == "categorize":
            reward, feedback = self._handle_categorize(action)
        elif action_type == "rank":
            reward, feedback = self._handle_rank(action)
        elif action_type == "recommend":
            reward, feedback = self._handle_recommend(action)
            done = True

        # Auto-end if max steps reached
        if self._state.step_count >= max_steps and not done:
            done = True
            # If no recommendation was made, auto-recommend from last ranking or pool
            if not self._recommended_ids:
                k = self._task_config["recommend_k"]
                if self._last_ranking:
                    self._recommended_ids = self._last_ranking[:k]
                else:
                    pool_ids = [it["id"] for it in self._current_pool]
                    self._recommended_ids = pool_ids[:k]
                # Compute final episode score
                reward = self._compute_final_score()
                feedback = ActionFeedback(
                    relevance_score=reward,
                    explanation="Episode ended (max steps). Auto-recommended from best available ranking.",
                )

        return self._make_observation(reward=reward, done=done, feedback=feedback)

    @property
    def state(self) -> State:
        return self._state

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _handle_filter(self, action: CuratorAction) -> tuple[float, ActionFeedback]:
        """Remove items from the pool. Reward for removing low-relevance items."""
        valid_ids = [iid for iid in action.item_ids if iid in self._items_by_id]
        if not valid_ids:
            return 0.0, ActionFeedback(explanation="No valid items to filter.")

        # Remove from pool
        for iid in valid_ids:
            self._items_by_id.pop(iid, None)
            self._filtered_ids.append(iid)
        self._current_pool = [
            it for it in self._current_pool if it["id"] in self._items_by_id
        ]
        self._items_filtered_count += len(valid_ids)

        # Score: reward for removing low-relevance items
        quality = grader.filter_quality(valid_ids, self._relevance)

        return quality, ActionFeedback(
            relevance_score=quality,
            explanation=f"Filtered {len(valid_ids)} items. Quality={quality:.3f}",
        )

    def _handle_categorize(
        self, action: CuratorAction
    ) -> tuple[float, ActionFeedback]:
        """Categorize items. Reward for matching relevance-derived categories."""
        if not action.categories:
            return 0.0, ActionFeedback(explanation="No categories provided.")

        valid_cats = {
            iid: cat
            for iid, cat in action.categories.items()
            if iid in self._items_by_id
        }
        if not valid_cats:
            return 0.0, ActionFeedback(explanation="No valid items to categorize.")

        self._categories.update(valid_cats)
        self._items_categorized_count += len(valid_cats)

        quality = grader.categorize_quality(valid_cats, self._relevance)

        return quality, ActionFeedback(
            relevance_score=quality,
            explanation=f"Categorized {len(valid_cats)} items. Accuracy={quality:.3f}",
        )

    def _handle_rank(self, action: CuratorAction) -> tuple[float, ActionFeedback]:
        """Rank items by priority. Reward based on NDCG."""
        rankings = action.rankings or action.item_ids
        if not rankings:
            return 0.0, ActionFeedback(explanation="No ranking provided.")

        valid_ranking = [iid for iid in rankings if iid in self._items_by_id]
        self._last_ranking = valid_ranking

        k = self._task_config["recommend_k"]
        quality = grader.ndcg_at_k(valid_ranking, self._relevance, k)

        # Also compute coverage
        coverage = grader.source_diversity(valid_ranking[:k], self._items_by_id)

        return quality, ActionFeedback(
            relevance_score=quality,
            coverage_score=coverage,
            explanation=f"Ranked {len(valid_ranking)} items. NDCG@{k}={quality:.3f}",
        )

    def _handle_recommend(
        self, action: CuratorAction
    ) -> tuple[float, ActionFeedback]:
        """Final recommendation. Triggers episode end with composite score."""
        rec_ids = action.item_ids
        k = self._task_config["recommend_k"]

        if not rec_ids:
            # Fall back to last ranking
            if self._last_ranking:
                rec_ids = self._last_ranking[:k]
            else:
                return 0.0, ActionFeedback(
                    explanation="No items recommended and no prior ranking."
                )

        self._recommended_ids = rec_ids[:k]
        score = self._compute_final_score()

        return score, ActionFeedback(
            relevance_score=score,
            coverage_score=grader.source_diversity(
                self._recommended_ids, self._items_by_id
            ),
            explanation=f"Final recommendation of {len(self._recommended_ids)} items. Score={score:.3f}",
        )

    # =========================================================================
    # Scoring
    # =========================================================================

    def _compute_final_score(self) -> float:
        """Compute composite episode score."""
        return grader.grade_episode(
            recommended_ids=self._recommended_ids,
            ranked_ids=self._last_ranking if self._last_ranking else None,
            categories=self._categories if self._categories else None,
            relevance_scores=self._relevance,
            items_by_id=self._items_by_id,
            recommend_k=self._task_config["recommend_k"],
        )

    # =========================================================================
    # Observation Builder
    # =========================================================================

    def _make_observation(
        self,
        reward: float,
        done: bool,
        feedback: Optional[ActionFeedback] = None,
    ) -> CuratorObservation:
        items = [ContentItem(**it) for it in self._current_pool]
        profile = UserProfile(**self._profile) if self._profile else None

        task_info = None
        if self._task_config:
            task_info = TaskInfo(
                task_id=self._task_config["task_id"],
                difficulty=self._task_config["difficulty"],
                max_steps=self._task_config["max_steps"],
                recommend_k=self._task_config["recommend_k"],
                pool_size=len(self._current_pool),
                items_filtered=self._items_filtered_count,
                items_categorized=self._items_categorized_count,
                step_number=self._state.step_count,
            )

        return CuratorObservation(
            items=items,
            user_profile=profile,
            feedback=feedback,
            task_info=task_info,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
            },
        )
