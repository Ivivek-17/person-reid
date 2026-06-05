from pathlib import Path

import faiss
import numpy as np

from extract import PROJECT_ROOT, get_embedding, load_config, load_model


def search(query_path: str | Path, config: dict) -> None:
    index_path = Path(config["paths"]["index"])
    if not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path

    labels_path = PROJECT_ROOT / "embeddings" / "labels.npy"
    top_k = config["reid"]["top_k"]

    index = faiss.read_index(str(index_path))
    labels = np.load(labels_path)

    processor, model = load_model(config)
    query_embedding = get_embedding(query_path, processor, model)
    query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        print(f"Rank {rank}: {labels[idx]}, similarity: {score:.4f}")


if __name__ == "__main__":
    config = load_config()
    query_path = "data/Market_1501_dataset/query/0001_c1s1_001051_00.jpg"
    search(query_path, config)
