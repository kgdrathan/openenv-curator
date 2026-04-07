---
title: Curator Environment
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - OpenEnv
  - RL
---

# Curator — Personalized Content Curation Environment

An OpenEnv environment where an agent must curate a pool of real content items (from Hacker News, arXiv, DEV.to, Reddit) and curate a personalized reading list based on a user's preference profile.

## Goal

Every knowledge worker drowns in information — hundreds of articles, papers, and posts across dozens of sources daily. Given a user profile and a content pool, the agent must intelligently **filter**, **categorize**, **rank**, and **recommend** the most relevant items. Scored using standard Information Retrieval metrics (NDCG, precision, recall).

## Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `filter` | `item_ids: List[str]` | Remove irrelevant items from the pool |
| `categorize` | `categories: Dict[str, "urgent"\|"read_later"\|"share"\|"skip"]` | Tag items by priority |
| `rank` | `rankings: List[str]` | Order items by relevance (best first) |
| `recommend` | `item_ids: List[str]` | Final recommendation (ends episode) |

## Observation Space

Each observation includes:

- **items** — current pool of content items (`id`, `source`, `title`, `summary`, `tags`, `score`, `reading_time_mins`, `content_type`)
- **user_profile** — interests (topic weights 0-1), preferred sources, skill level, time budget, read history
- **feedback** — per-step scores (relevance, coverage) from the last action
- **task_info** — difficulty, max steps, progress counters

## Tasks

| Task | Pool Size | Sources | Max Steps | Recommend K | Description |
|------|-----------|---------|-----------|-------------|-------------|
| **easy** | 20 | Hacker News | 10 | 5 | Clear AI/ML interests, single source |
| **medium** | 50 | HN + arXiv + DEV.to | 20 | 10 | Broad interests, 3 sources, some already-read items |
| **hard** | 100 | All 4 sources | 30 | 15 | Minimal preferences, must infer interests from feedback |

Each task includes an embedded user profile that defines what "relevant" means for scoring.

## Scoring

**Per-step rewards** (0-1):
- **filter**: higher reward for removing low-relevance items
- **categorize**: accuracy against relevance-derived ground truth
- **rank**: NDCG@k against ground truth relevance
- **recommend**: composite final episode score

**Final episode score** (deterministic, 0-1):

```
score = 0.35 * NDCG@k + 0.25 * Precision@k + 0.20 * Recall@k + 0.10 * Category accuracy + 0.10 * Source diversity
```

## Data

All content is real data fetched from free public APIs (no auth needed), cached as static JSON — no API calls at runtime:

- **Hacker News** — top stories via Firebase API
- **arXiv** — recent AI/ML/NLP papers
- **DEV.to** — programming articles and tutorials
- **Reddit** — posts from r/programming, r/machinelearning, r/webdev
