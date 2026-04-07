"""
Grading module for Curator environment.

Implements standard Information Retrieval metrics for deterministic,
reproducible scoring of agent performance (0.0-1.0).
"""

import math
from typing import Dict, List, Optional


def dcg_at_k(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(
    ranked_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        ranked_ids: Agent's ranked list of item IDs (best first).
        relevance_scores: Ground truth {item_id: relevance} scores.
        k: Evaluate top-k items.

    Returns:
        NDCG score in [0, 1].
    """
    if not ranked_ids or not relevance_scores or k <= 0:
        return 0.0

    # Actual DCG from agent ranking
    actual_rels = [relevance_scores.get(iid, 0.0) for iid in ranked_ids[:k]]
    actual_dcg = dcg_at_k(actual_rels, k)

    # Ideal DCG (sorted by relevance, descending)
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_rels, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(
    selected_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int,
    threshold: float = 0.5,
) -> float:
    """Compute Precision at k.

    Args:
        selected_ids: Agent's selected item IDs.
        relevance_scores: Ground truth {item_id: relevance} scores.
        k: Evaluate top-k items.
        threshold: Minimum relevance to count as "relevant".

    Returns:
        Precision score in [0, 1].
    """
    if not selected_ids or k <= 0:
        return 0.0

    top_k = selected_ids[:k]
    relevant_count = sum(
        1 for iid in top_k if relevance_scores.get(iid, 0.0) >= threshold
    )
    return relevant_count / min(k, len(top_k))


def recall_at_k(
    selected_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int,
    threshold: float = 0.5,
) -> float:
    """Compute Recall at k.

    Args:
        selected_ids: Agent's selected item IDs.
        relevance_scores: Ground truth {item_id: relevance} scores.
        k: Evaluate top-k items.
        threshold: Minimum relevance to count as "relevant".

    Returns:
        Recall score in [0, 1].
    """
    total_relevant = sum(1 for v in relevance_scores.values() if v >= threshold)
    if total_relevant == 0:
        return 1.0  # No relevant items to find

    top_k = selected_ids[:k]
    found_relevant = sum(
        1 for iid in top_k if relevance_scores.get(iid, 0.0) >= threshold
    )
    return found_relevant / total_relevant


def source_diversity(selected_ids: List[str], items_by_id: Dict[str, dict]) -> float:
    """Compute source diversity of selected items.

    Returns:
        Diversity score in [0, 1] based on unique source coverage.
    """
    if not selected_ids:
        return 0.0

    all_sources = set(it.get("source", "") for it in items_by_id.values())
    selected_sources = set(
        items_by_id[iid].get("source", "") for iid in selected_ids if iid in items_by_id
    )
    if not all_sources:
        return 0.0
    return len(selected_sources) / len(all_sources)


def filter_quality(
    removed_ids: List[str],
    relevance_scores: Dict[str, float],
) -> float:
    """Score a filter action: reward for removing low-relevance items.

    Returns:
        Score in [0, 1]. Higher is better (removed less relevant items).
    """
    if not removed_ids:
        return 0.0

    avg_relevance_of_removed = sum(
        relevance_scores.get(iid, 0.5) for iid in removed_ids
    ) / len(removed_ids)

    # Good filtering removes low-relevance items
    return max(0.0, min(1.0, 1.0 - avg_relevance_of_removed))


def categorize_quality(
    agent_categories: Dict[str, str],
    relevance_scores: Dict[str, float],
    threshold_urgent: float = 0.7,
    threshold_read: float = 0.4,
) -> float:
    """Score categorization accuracy against relevance-derived ground truth.

    Ground truth categories derived from relevance:
        >= threshold_urgent → "urgent"
        >= threshold_read   → "read_later"
        < threshold_read    → "skip"
        (any relevance can be "share" — not penalized)

    Returns:
        Accuracy score in [0, 1].
    """
    if not agent_categories:
        return 0.0

    correct = 0
    total = len(agent_categories)

    for item_id, agent_cat in agent_categories.items():
        rel = relevance_scores.get(item_id, 0.0)

        # Derive expected category
        if rel >= threshold_urgent:
            expected = {"urgent", "share"}
        elif rel >= threshold_read:
            expected = {"read_later", "share"}
        else:
            expected = {"skip"}

        if agent_cat in expected:
            correct += 1

    return correct / total


def grade_episode(
    recommended_ids: List[str],
    ranked_ids: Optional[List[str]],
    categories: Optional[Dict[str, str]],
    relevance_scores: Dict[str, float],
    items_by_id: Dict[str, dict],
    recommend_k: int,
) -> float:
    """Compute final episode score (0-1).

    Composite:
        0.35 * NDCG@k
        0.25 * Precision@k
        0.20 * Recall@k
        0.10 * Category accuracy
        0.10 * Source diversity
    """
    # Use recommended_ids as ranking if no explicit ranking
    ranking = ranked_ids if ranked_ids else recommended_ids

    ndcg = ndcg_at_k(ranking, relevance_scores, recommend_k)
    precision = precision_at_k(recommended_ids, relevance_scores, recommend_k)
    recall = recall_at_k(recommended_ids, relevance_scores, recommend_k)
    cat_acc = categorize_quality(categories, relevance_scores) if categories else 0.0
    diversity = source_diversity(recommended_ids, items_by_id)

    score = (
        0.35 * ndcg
        + 0.25 * precision
        + 0.20 * recall
        + 0.10 * cat_acc
        + 0.10 * diversity
    )
    return max(0.0, min(1.0, score))
