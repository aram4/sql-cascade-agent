"""Download and prepare a slice of the BIRD benchmark."""

import json
import os

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bird")


def download_bird() -> list[dict]:
    """Download BIRD dev set from HuggingFace and cache locally."""
    cache_path = os.path.join(DATA_DIR, "bird_slice.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    print("Downloading BIRD benchmark from HuggingFace...")
    ds = load_dataset("xu3kev/BIRD-SQL-data", split="train")

    items = []
    for i, item in enumerate(ds):
        items.append({
            "question_id": i,
            "question": item["question"],
            "gold_sql": item["SQL"],
            "db_id": item["db_id"],
            "evidence": item.get("evidence", ""),
            "schema": item["schema"],
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"Saved {len(items)} questions to {cache_path}")
    return items


if __name__ == "__main__":
    items = download_bird()
    dbs = set(item["db_id"] for item in items)
    print(f"Dataset has {len(items)} questions across {len(dbs)} databases")
    print(f"Databases: {sorted(dbs)}")
