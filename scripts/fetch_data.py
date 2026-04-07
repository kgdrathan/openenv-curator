#!/usr/bin/env python3
"""
Fetch real content items from public APIs and save as static JSON.

Sources (all free, no auth):
- Hacker News (Firebase API)
- arXiv (public API)
- DEV.to (public API)
- Reddit (public JSON)

Run once: python scripts/fetch_data.py
Output: data/items.json
"""

import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data"

# Tag extraction keywords
TAG_KEYWORDS = {
    "ai": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml",
        "deep learning",
        "neural",
    ],
    "nlp": [
        "nlp",
        "natural language",
        "language model",
        "llm",
        "gpt",
        "transformer",
        "bert",
    ],
    "web": [
        "web",
        "javascript",
        "react",
        "frontend",
        "css",
        "html",
        "browser",
        "nextjs",
        "vue",
    ],
    "systems": [
        "systems",
        "linux",
        "kernel",
        "os",
        "distributed",
        "infrastructure",
        "devops",
    ],
    "rust": ["rust", "cargo", "rustc", "borrow checker"],
    "python": ["python", "pip", "django", "flask", "fastapi", "pytorch"],
    "go": ["golang", " go ", "goroutine"],
    "security": [
        "security",
        "vulnerability",
        "exploit",
        "crypto",
        "encryption",
        "privacy",
    ],
    "database": ["database", "sql", "postgres", "mongodb", "redis", "sqlite"],
    "cloud": ["cloud", "aws", "gcp", "azure", "kubernetes", "docker", "k8s"],
    "mobile": ["mobile", "ios", "android", "swift", "kotlin", "flutter"],
    "data": [
        "data",
        "analytics",
        "visualization",
        "pandas",
        "spark",
        "etl",
        "pipeline",
    ],
    "career": ["career", "hiring", "interview", "salary", "remote", "job"],
    "startup": ["startup", "funding", "venture", "entrepreneur", "saas", "product"],
    "open-source": [
        "open source",
        "open-source",
        "oss",
        "github",
        "foss",
        "mit license",
    ],
    "robotics": ["robot", "robotics", "autonomous", "drone", "perception", "slam"],
    "cv": ["computer vision", "image", "object detection", "segmentation", "diffusion"],
}


def extract_tags(title: str, summary: str = "") -> list[str]:
    """Extract topic tags from title and summary text."""
    text = f"{title} {summary}".lower()
    tags = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)
    return tags if tags else ["general"]


def fetch_json(url: str, headers: dict | None = None) -> dict | list:
    """Fetch JSON from a URL."""
    req = Request(url, headers=headers or {"User-Agent": "Curator/1.0"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_text(url: str) -> str:
    """Fetch raw text from a URL."""
    req = Request(url, headers={"User-Agent": "Curator/1.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def fetch_hackernews(count: int = 60) -> list[dict]:
    """Fetch top stories from Hacker News."""
    print(f"  Fetching {count} Hacker News stories...")
    story_ids = fetch_json("https://hacker-news.firebaseio.com/v0/topstories.json")
    items = []
    for sid in story_ids[:count]:
        try:
            story = fetch_json(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
            if not story or story.get("type") != "story":
                continue
            title = story.get("title", "")
            url = story.get("url", f"https://news.ycombinator.com/item?id={sid}")
            items.append(
                {
                    "id": f"hn_{sid}",
                    "source": "hackernews",
                    "title": title,
                    "summary": title,  # HN doesn't have summaries; title is the content
                    "tags": extract_tags(title),
                    "url": url,
                    "author": story.get("by", ""),
                    "score": story.get("score", 0),
                    "reading_time_mins": 5,
                    "content_type": "article",
                }
            )
        except Exception as e:
            print(f"    Skipping HN story {sid}: {e}")
        time.sleep(0.05)  # Be polite
    print(f"    Got {len(items)} HN items")
    return items


def fetch_arxiv(count: int = 50) -> list[dict]:
    """Fetch recent AI/ML papers from arXiv."""
    print(f"  Fetching {count} arXiv papers...")
    categories = "cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"
    url = f"https://export.arxiv.org/api/query?search_query={categories}&sortBy=submittedDate&sortOrder=descending&max_results={count}"
    xml_text = fetch_text(url)
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    items = []
    for entry in root.findall("atom:entry", ns):
        try:
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = (
                entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:300]
            )
            authors = [
                a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)
            ]
            link = entry.find("atom:id", ns).text

            # Estimate reading time from summary length
            word_count = len(summary.split())
            reading_time = max(10, word_count // 20)

            items.append(
                {
                    "id": f"arxiv_{arxiv_id.replace('/', '_').replace('.', '_')}",
                    "source": "arxiv",
                    "title": title,
                    "summary": summary,
                    "tags": extract_tags(title, summary),
                    "url": link,
                    "author": authors[0] if authors else "",
                    "score": 0,
                    "reading_time_mins": reading_time,
                    "content_type": "paper",
                }
            )
        except Exception as e:
            print(f"    Skipping arXiv entry: {e}")

    print(f"    Got {len(items)} arXiv items")
    return items


def fetch_devto(count: int = 50) -> list[dict]:
    """Fetch articles from DEV.to."""
    print(f"  Fetching {count} DEV.to articles...")
    items = []
    # Fetch from multiple tags to get variety
    tags_to_fetch = ["programming", "ai", "webdev", "python", "tutorial"]
    per_tag = math.ceil(count / len(tags_to_fetch))

    seen_ids = set()
    for tag in tags_to_fetch:
        try:
            articles = fetch_json(
                f"https://dev.to/api/articles?per_page={per_tag}&tag={tag}&top=7"
            )
            for article in articles:
                aid = article["id"]
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                title = article.get("title", "")
                desc = article.get("description", "")
                tag_list = article.get("tag_list", [])
                items.append(
                    {
                        "id": f"devto_{aid}",
                        "source": "devto",
                        "title": title,
                        "summary": desc[:300] if desc else title,
                        "tags": extract_tags(title, desc)
                        if not tag_list
                        else [t.lower() for t in tag_list[:5]],
                        "url": article.get("url", ""),
                        "author": article.get("user", {}).get("username", ""),
                        "score": article.get("positive_reactions_count", 0),
                        "reading_time_mins": article.get("reading_time_minutes", 5),
                        "content_type": "tutorial"
                        if "tutorial" in (tag_list or [])
                        else "article",
                    }
                )
            time.sleep(0.2)
        except Exception as e:
            print(f"    Skipping DEV.to tag {tag}: {e}")

    items = items[:count]
    print(f"    Got {len(items)} DEV.to items")
    return items


def fetch_reddit(count: int = 40) -> list[dict]:
    """Fetch posts from Reddit programming subreddits."""
    print(f"  Fetching {count} Reddit posts...")
    items = []
    subreddits = ["programming", "machinelearning", "webdev"]
    per_sub = math.ceil(count / len(subreddits))

    seen_ids = set()
    for sub in subreddits:
        try:
            data = fetch_json(
                f"https://www.reddit.com/r/{sub}/hot.json?limit={per_sub}",
                headers={"User-Agent": "Curator/1.0 (content-curation-research)"},
            )
            for post in data.get("data", {}).get("children", []):
                pd = post["data"]
                rid = pd["id"]
                if rid in seen_ids or pd.get("stickied"):
                    continue
                seen_ids.add(rid)
                title = pd.get("title", "")
                selftext = pd.get("selftext", "")[:300]
                items.append(
                    {
                        "id": f"reddit_{rid}",
                        "source": "reddit",
                        "title": title,
                        "summary": selftext if selftext else title,
                        "tags": extract_tags(title, selftext),
                        "url": f"https://reddit.com{pd.get('permalink', '')}",
                        "author": pd.get("author", ""),
                        "score": pd.get("score", 0),
                        "reading_time_mins": max(2, len(selftext.split()) // 200)
                        if selftext
                        else 3,
                        "content_type": "discussion",
                    }
                )
            time.sleep(0.5)
        except Exception as e:
            print(f"    Skipping Reddit r/{sub}: {e}")

    items = items[:count]
    print(f"    Got {len(items)} Reddit items")
    return items


def compute_relevance(item: dict, profile: dict) -> float:
    """Compute relevance score (0-1) of an item for a user profile.

    Scoring:
    - 0.50 weight: tag match (sum of matched interest weights / total interest weight)
    - 0.20 weight: source preference (1.0 if preferred, 0.3 otherwise)
    - 0.15 weight: community signal (normalized score/upvotes)
    - 0.10 weight: reading time fit (within budget = 1.0, over = 0.3)
    - 0.05 weight: content type match (paper for expert, tutorial for beginner)
    - Penalty: -0.4 for already-read items
    """
    interests = profile["interests"]
    item_tags = set(item["tags"])

    if not interests:
        return 0.05

    # Tag match: how much of the user's interest space does this item cover?
    total_interest_weight = sum(interests.values())
    matched_weight = sum(interests.get(tag, 0.0) for tag in item_tags)
    tag_score = matched_weight / total_interest_weight if total_interest_weight > 0 else 0.0

    # Source preference
    preferred = profile.get("preferred_sources", [])
    source_score = 1.0 if (not preferred or item["source"] in preferred) else 0.3

    # Community signal (normalize score: 0-100+ -> 0-1)
    raw_score = item.get("score", 0)
    community_score = min(1.0, raw_score / 200) if raw_score > 0 else 0.2

    # Reading time fit
    budget = profile.get("time_budget_mins", 60)
    per_item_budget = budget / 5
    time_score = 1.0 if item["reading_time_mins"] <= per_item_budget else 0.3

    # Content type match
    skill = profile.get("skill_level", "intermediate")
    ctype = item.get("content_type", "article")
    if skill == "expert" and ctype == "paper":
        type_score = 1.0
    elif skill == "beginner" and ctype in ("tutorial", "article"):
        type_score = 1.0
    elif skill == "intermediate":
        type_score = 0.8
    else:
        type_score = 0.5

    # Weighted combination
    relevance = (
        0.50 * tag_score
        + 0.20 * source_score
        + 0.15 * community_score
        + 0.10 * time_score
        + 0.05 * type_score
    )

    # Already-read penalty
    if item["id"] in profile.get("read_history", []):
        relevance -= 0.4

    return round(max(0.0, min(1.0, relevance)), 4)


def create_tasks() -> list[dict]:
    """Create task definitions with embedded user profiles for 3 difficulty levels."""
    return [
        {
            "task_id": "easy",
            "difficulty": "easy",
            "item_count": 20,
            "max_steps": 10,
            "sources": ["hackernews"],
            "recommend_k": 5,
            "description": "Curate 5 top articles from 20 Hacker News stories for an AI/ML enthusiast.",
            "profile": {
                "interests": {
                    "ai": 0.95,
                    "nlp": 0.85,
                    "python": 0.8,
                    "data": 0.7,
                },
                "preferred_sources": ["hackernews", "arxiv"],
                "time_budget_mins": 120,
                "read_history": [],
                "skill_level": "intermediate",
            },
        },
        {
            "task_id": "medium",
            "difficulty": "medium",
            "item_count": 50,
            "max_steps": 20,
            "sources": ["hackernews", "devto", "arxiv"],
            "recommend_k": 10,
            "description": "Curate 10 items from 50 across HN, DEV.to, and arXiv for a senior engineer with broad interests.",
            "profile": {
                "interests": {
                    "ai": 0.9,
                    "web": 0.7,
                    "systems": 0.6,
                    "security": 0.5,
                    "python": 0.75,
                    "cloud": 0.4,
                    "open-source": 0.65,
                    "startup": 0.3,
                },
                "preferred_sources": ["hackernews", "devto"],
                "time_budget_mins": 60,
                "read_history": [],
                "skill_level": "expert",
            },
        },
        {
            "task_id": "hard",
            "difficulty": "hard",
            "item_count": 100,
            "max_steps": 30,
            "sources": ["hackernews", "devto", "arxiv", "reddit"],
            "recommend_k": 15,
            "description": "Curate 15 items from 100 across all sources for a beginner with minimal stated preferences. Must infer interests from feedback.",
            "profile": {
                "interests": {
                    "rust": 0.5,
                    "systems": 0.4,
                },
                "preferred_sources": [],
                "time_budget_mins": 30,
                "read_history": [],
                "skill_level": "beginner",
            },
        },
    ]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("Fetching real content data from public APIs...\n")

    # Fetch from all sources
    all_items = []
    all_items.extend(fetch_hackernews(60))
    all_items.extend(fetch_arxiv(50))
    all_items.extend(fetch_devto(50))
    all_items.extend(fetch_reddit(40))

    print(f"\nTotal items fetched: {len(all_items)}")

    # Save items
    items_path = DATA_DIR / "items.json"
    with open(items_path, "w") as f:
        json.dump(all_items, f, indent=2)
    print(f"Saved items to {items_path}")

    # Create tasks (profiles are embedded in each task)
    tasks = create_tasks()

    # Compute ground truth relevance and set read_history
    ground_truth = {}
    for task in tasks:
        profile = task["profile"]
        sources = task["sources"]
        task_items = [it for it in all_items if it["source"] in sources][
            : task["item_count"]
        ]

        # Set some items as already read for medium/hard tasks
        if task["task_id"] == "medium" and len(task_items) > 5:
            profile["read_history"] = [task_items[i]["id"] for i in range(0, 6, 2)]
        elif task["task_id"] == "hard" and len(task_items) > 10:
            profile["read_history"] = [task_items[i]["id"] for i in range(0, 10, 3)]

        relevance = {}
        for item in task_items:
            relevance[item["id"]] = round(compute_relevance(item, profile), 4)
        ground_truth[task["task_id"]] = relevance

    # Save tasks (with updated read_history in profiles)
    tasks_path = DATA_DIR / "tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved tasks to {tasks_path}")

    gt_path = DATA_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground truth to {gt_path}")

    # Summary
    print("\n--- Summary ---")
    for task in tasks:
        tid = task["task_id"]
        gt = ground_truth[tid]
        avg_rel = sum(gt.values()) / len(gt) if gt else 0
        high_rel = sum(1 for v in gt.values() if v >= 0.5)
        print(
            f"  {tid}: {len(gt)} items, avg relevance={avg_rel:.3f}, high-relevance={high_rel}"
        )


if __name__ == "__main__":
    main()
