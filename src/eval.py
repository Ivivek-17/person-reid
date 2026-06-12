from pathlib import Path

import numpy as np
from tqdm import tqdm

from extract import PROJECT_ROOT, load_config, load_model
from search import load_index_and_labels, search_matches


def get_person_id(filename: str) -> str:
    return filename[:4]


def load_query_embeddings(config: dict):
    cache_dir = Path(config["paths"].get("query_embeddings", ""))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    embeddings_path = cache_dir / "embeddings.npy"
    labels_path = cache_dir / "labels.npy"

    if embeddings_path.exists() and labels_path.exists():
        embeddings = np.load(embeddings_path).astype(np.float32)
        filenames = np.load(labels_path)
        return embeddings, filenames.tolist()

    return None, None


def average_precision(matches: list[tuple[str, float]], query_id: str) -> float:
    hits = 0
    precision_sum = 0.0
    total_relevant = sum(1 for filename, _ in matches if get_person_id(filename) == query_id)

    if total_relevant == 0:
        return 0.0

    for rank, (filename, _) in enumerate(matches, start=1):
        if get_person_id(filename) == query_id:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / total_relevant


def evaluate(config: dict) -> None:
    query_dir = Path(config["paths"]["query"])
    if not query_dir.is_absolute():
        query_dir = PROJECT_ROOT / query_dir

    precomputed_embeddings, precomputed_filenames = load_query_embeddings(config)
    use_precomputed = precomputed_embeddings is not None

    processor, model = load_model(config)
    index, labels = load_index_and_labels(config)

    query_paths = sorted(query_dir.glob("*.jpg"))
    if not query_paths:
        raise FileNotFoundError(f"No .jpg files found in {query_dir}")

    if use_precomputed:
        emb_lookup = {
            fname: precomputed_embeddings[i]
            for i, fname in enumerate(precomputed_filenames)
        }

    rank1_correct = 0
    rank5_correct = 0
    average_precisions = []
    for query_path in tqdm(query_paths, desc="Evaluating"):
        query_id = get_person_id(query_path.name)

        if use_precomputed and query_path.name in emb_lookup:
            embedding = emb_lookup[query_path.name]
            matches = search_matches(
                query_path, processor, model, index, labels,
                top_k=10, precomputed_embedding=embedding
            )
        else:
            matches = search_matches(
                query_path, processor, model, index, labels, top_k=10
            )

        top1_filename = matches[0][0]
        if top1_filename.startswith(query_id):
            rank1_correct += 1

        if any(get_person_id(filename) == query_id for filename, _ in matches[:5]):
            rank5_correct += 1

        average_precisions.append(average_precision(matches, query_id))

    rank1 = rank1_correct / len(query_paths) * 100
    rank5 = rank5_correct / len(query_paths) * 100
    mean_ap = float(np.mean(average_precisions)) * 100
    print("\nEvaluation Metrics")
    print("------------------")
    print(f"Rank-1: {rank1:.2f}% ({rank1_correct}/{len(query_paths)})")
    print(f"Rank-5: {rank5:.2f}% ({rank5_correct}/{len(query_paths)})")
    print(f"mAP:    {mean_ap:.2f}%")


if __name__ == "__main__":
    config = load_config()
    evaluate(config)
