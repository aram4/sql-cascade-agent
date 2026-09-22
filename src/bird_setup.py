"""Download BIRD benchmark: questions locally, databases to a Modal Volume."""

import json
import os
import random

import modal
from huggingface_hub import hf_hub_download, HfApi

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bird")
HF_REPO = "premai-io/birdbench"
VOLUME_NAME = "bird-databases"
SLICE_SIZE = 250

app = modal.App("bird-setup")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim().pip_install("huggingface_hub")


def download_questions() -> list[dict]:
    """Download validation.json and create a stratified 250-question slice."""
    cache_path = os.path.join(DATA_DIR, "bird_slice.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="validation/validation.json",
        repo_type="dataset",
        local_dir=DATA_DIR,
    )

    with open(local_path) as f:
        all_questions = json.load(f)

    for i, q in enumerate(all_questions):
        q["question_id"] = i

    random.seed(42)
    if len(all_questions) > SLICE_SIZE:
        db_groups = {}
        for q in all_questions:
            db_groups.setdefault(q["db_id"], []).append(q)

        sliced = []
        per_db = max(1, SLICE_SIZE // len(db_groups))
        for db_id, questions in db_groups.items():
            sliced.extend(random.sample(questions, min(per_db, len(questions))))

        while len(sliced) < SLICE_SIZE:
            remaining = [q for q in all_questions if q not in sliced]
            if not remaining:
                break
            sliced.append(random.choice(remaining))

        sliced = sliced[:SLICE_SIZE]
    else:
        sliced = all_questions

    with open(cache_path, "w") as f:
        json.dump(sliced, f, indent=2)

    print(f"Saved {len(sliced)} questions to {cache_path}")
    dbs = set(q["db_id"] for q in sliced)
    print(f"Covers {len(dbs)} databases: {sorted(dbs)}")
    return sliced


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def upload_database(db_id: str):
    """Download a BIRD database from HuggingFace and save to Modal Volume."""
    from huggingface_hub import hf_hub_download
    import shutil

    dest = f"/data/{db_id}/{db_id}.sqlite"
    if os.path.exists(dest):
        print(f"  {db_id}: already exists, skipping")
        return

    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"validation/dev_databases/{db_id}/{db_id}.sqlite",
        repo_type="dataset",
    )

    os.makedirs(f"/data/{db_id}", exist_ok=True)
    shutil.copy2(local_path, dest)
    volume.commit()
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  {db_id}: uploaded ({size_mb:.1f} MB)")


@app.local_entrypoint()
def main():
    print("Step 1: Downloading questions...")
    questions = download_questions()

    db_ids = sorted(set(q["db_id"] for q in questions))
    print(f"\nStep 2: Uploading {len(db_ids)} databases to Modal Volume '{VOLUME_NAME}'...")

    for result in upload_database.map(db_ids):
        pass

    print("\nDone! Databases are on Modal Volume, questions are local.")
