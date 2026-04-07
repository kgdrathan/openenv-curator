"""Curator Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    ActionFeedback,
    ContentItem,
    CuratorAction,
    CuratorObservation,
    TaskInfo,
    UserProfile,
)


class CuratorEnv(EnvClient[CuratorAction, CuratorObservation, State]):
    """
    Client for the Curator Environment.

    Example:
        >>> async with CuratorEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset(task_id="easy")
        ...     print(len(result.observation.items))
        ...
        ...     result = await client.step(CuratorAction(
        ...         action_type="rank",
        ...         rankings=["hn_123", "hn_456"]
        ...     ))

    Example with Docker:
        >>> client = await CuratorEnv.from_docker_image("curator:latest")
    """

    def _step_payload(self, action: CuratorAction) -> Dict:
        """Convert CuratorAction to JSON payload."""
        payload = {"action_type": action.action_type}

        if action.item_ids:
            payload["item_ids"] = action.item_ids
        if action.categories is not None:
            payload["categories"] = action.categories
        if action.rankings is not None:
            payload["rankings"] = action.rankings
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        if action.metadata:
            payload["metadata"] = action.metadata

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[CuratorObservation]:
        """Parse server response into StepResult[CuratorObservation]."""
        obs_data = payload.get("observation", {})

        # Parse nested models
        items = [ContentItem(**it) for it in obs_data.get("items", [])]

        user_profile = None
        if obs_data.get("user_profile"):
            user_profile = UserProfile(**obs_data["user_profile"])

        feedback = None
        if obs_data.get("feedback"):
            feedback = ActionFeedback(**obs_data["feedback"])

        task_info = None
        if obs_data.get("task_info"):
            task_info = TaskInfo(**obs_data["task_info"])

        observation = CuratorObservation(
            items=items,
            user_profile=user_profile,
            feedback=feedback,
            task_info=task_info,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
