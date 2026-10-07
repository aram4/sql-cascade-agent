"""Trains the router classifier: given a question + schema, predict whether the
cheap `generate` model is sufficient, or whether the question should be escalated
to the strong model.

Label convention: needs_big = 1 only when the small model failed AND the large
model succeeded on that question — i.e., escalating demonstrably helps. If the
small model already succeeds, or if both fail, there's no case for paying for
the big model, so needs_big = 0 in both those situations. This directly targets
the cascade's actual goal (skip the big model unless it provably changes the
outcome), not just "which model is generally more accurate."
"""

import json

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.router_features import FEATURE_NAMES, extract_features

BIRD_SLICE_PATH = "data/bird/bird_slice.json"
SMALL_RESULTS_PATH = "data/bird_eval_results_small.json"
LARGE_RESULTS_PATH = "data/bird_eval_results_large.json"
SCHEMA_CACHE_PATH = "data/bird_schemas_cache.json"
MODEL_OUT_PATH = "data/router_classifier.joblib"


def build_dataset() -> tuple[list[dict], list[int], list[dict]]:
    questions = {q["question_id"]: q for q in json.load(open(BIRD_SLICE_PATH))}
    small = {r["question_id"]: r["match"] for r in json.load(open(SMALL_RESULTS_PATH))["results"]}
    large = {r["question_id"]: r["match"] for r in json.load(open(LARGE_RESULTS_PATH))["results"]}
    schemas = json.load(open(SCHEMA_CACHE_PATH))

    rows, labels, meta = [], [], []
    for qid, q in questions.items():
        if qid not in small or qid not in large:
            continue
        small_ok, large_ok = small[qid], large[qid]
        needs_big = int((not small_ok) and large_ok)
        feats = extract_features(q, schemas[q["db_id"]])
        rows.append(feats)
        labels.append(needs_big)
        meta.append({"question_id": qid, "db_id": q["db_id"], "small_ok": small_ok, "large_ok": large_ok})

    return rows, labels, meta


def main():
    rows, labels, meta = build_dataset()
    print(f"Dataset: {len(rows)} questions")
    print(f"  needs_big=1 (small failed, big succeeded): {sum(labels)}")
    print(f"  needs_big=0 (small sufficient, or both failed): {len(labels) - sum(labels)}")

    X = [[r[name] for name in FEATURE_NAMES] for r in rows]
    y = labels

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    best_name, best_model, best_acc = None, None, -1
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"Test accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, target_names=["small_sufficient", "needs_big"], zero_division=0))
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest model: {best_name} ({best_acc:.3f} test accuracy)")

    if hasattr(best_model, "feature_importances_"):
        importances = sorted(zip(FEATURE_NAMES, best_model.feature_importances_), key=lambda x: -x[1])
        print("\nFeature importances:")
        for name, imp in importances:
            print(f"  {name:<28} {imp:.3f}")
    elif hasattr(best_model, "coef_"):
        importances = sorted(zip(FEATURE_NAMES, best_model.coef_[0]), key=lambda x: -abs(x[1]))
        print("\nCoefficients:")
        for name, coef in importances:
            print(f"  {name:<28} {coef:+.3f}")

    joblib.dump({"model": best_model, "feature_names": FEATURE_NAMES}, MODEL_OUT_PATH)
    print(f"\nSaved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
