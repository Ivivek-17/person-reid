from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from extract import PROJECT_ROOT, get_embedding, load_config, load_model


def build_index(config: dict) -> None:
    gallery_dir = Path(config["paths"]["gallery"])
    if not gallery_dir.is_absolute():
        gallery_dir = PROJECT_ROOT / gallery_dir

    index_path = Path(config["paths"]["index"])
    if not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path

    embedding_dim = config["model"]["embedding_dim"]
    labels_path = PROJECT_ROOT / "embeddings" / "labels.npy"

    processor, model = load_model(config)

    image_paths = sorted(gallery_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found in {gallery_dir}")

    embeddings = []
    labels = []

    for image_path in tqdm(image_paths, desc="Extracting embeddings"):
        embeddings.append(get_embedding(image_path, processor, model))
        labels.append(image_path.name)

    embeddings = np.stack(embeddings).astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(labels_path, np.array(labels))


if __name__ == "__main__":
    config = load_config()
    build_index(config)
