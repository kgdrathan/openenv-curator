"""
Data models for the Curator Environment.

Curator is a personalized content curation environment where an agent
must filter, categorize, rank, and recommend content items from a mixed
pool of real articles across multiple sources.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

# =============================================================================
# Helper Models
# =============================================================================


class ContentItem(BaseModel):
    """A single content item from any source."""

    id: str = Field(..., description="Unique item identifier")
    source: str = Field(
        ..., description="Content source: hackernews, arxiv, devto, reddit"
    )
    title: str = Field(..., description="Item title")
    summary: str = Field(default="", description="Brief summary or description")
    tags: List[str] = Field(default_factory=list, description="Topic tags")
    url: str = Field(default="", description="Original URL")
    author: str = Field(default="", description="Author name")
    score: int = Field(default=0, description="Community score/upvotes")
    reading_time_mins: int = Field(default=5, description="Estimated reading time")
    content_type: str = Field(
        default="article",
        description="Type: article, paper, discussion, job, tutorial, event",
    )


class UserProfile(BaseModel):
    """A user's preference profile for content curation."""

    interests: Dict[str, float] = Field(
        ..., description="Topic interest weights (0.0-1.0)"
    )
    preferred_sources: List[str] = Field(
        default_factory=list, description="Preferred content sources"
    )
    time_budget_mins: int = Field(
        default=60, description="Available reading time in minutes"
    )
    read_history: List[str] = Field(
        default_factory=list, description="IDs of already-read items"
    )
    skill_level: str = Field(
        default="intermediate",
        description="User expertise: beginner, intermediate, expert",
    )


class ActionFeedback(BaseModel):
    """Feedback from the environment after an action."""

    relevance_score: float = Field(
        default=0.0, description="How relevant the action's items were (0-1)"
    )
    coverage_score: float = Field(
        default=0.0, description="Source/topic diversity score (0-1)"
    )
    redundancy_penalty: float = Field(
        default=0.0, description="Penalty for recommending already-seen items (0-1)"
    )
    explanation: str = Field(default="", description="Explanation of the feedback")


class TaskInfo(BaseModel):
    """Information about the current task configuration."""

    task_id: str = Field(..., description="Task identifier: easy, medium, hard")
    difficulty: str = Field(..., description="Difficulty level")
    max_steps: int = Field(..., description="Maximum steps allowed")
    recommend_k: int = Field(..., description="Number of items to recommend")
    pool_size: int = Field(default=0, description="Current items in pool")
    items_filtered: int = Field(default=0, description="Items filtered so far")
    items_categorized: int = Field(default=0, description="Items categorized so far")
    step_number: int = Field(default=0, description="Current step number")


# =============================================================================
# Action & Observation Models
# =============================================================================


class CuratorAction(Action):
    """Action for the Curator environment.

    The agent can filter, categorize, rank, or recommend items.
    """

    action_type: Literal["filter", "categorize", "rank", "recommend"] = Field(
        ..., description="Type of action to perform"
    )
    item_ids: List[str] = Field(
        default_factory=list,
        description="Item IDs being acted on",
    )
    categories: Optional[
        Dict[str, Literal["urgent", "read_later", "share", "skip"]]
    ] = Field(
        default=None,
        description="Category assignments: {item_id: category} (for categorize action)",
    )
    rankings: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of item IDs by priority (for rank action)",
    )
    reasoning: Optional[str] = Field(
        default=None, description="Agent's reasoning for this action"
    )


class CuratorObservation(Observation):
    """Observation from the Curator environment."""

    items: List[ContentItem] = Field(
        default_factory=list, description="Current pool of content items"
    )
    user_profile: Optional[UserProfile] = Field(
        default=None, description="User preference profile"
    )
    feedback: Optional[ActionFeedback] = Field(
        default=None, description="Feedback from the last action"
    )
    task_info: Optional[TaskInfo] = Field(
        default=None, description="Current task configuration and progress"
    )
