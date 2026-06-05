from pathlib import Path

from tqdm import tqdm

from extract import PROJECT_ROOT, load_config, load_model
from search import load_index_and_labels, search_matches


def get_person_id(filename: str) -> str:
    return filename[:4]


def evaluate(config: dict) -> None:
    query_dir = Path(config["paths"]["query"])
    if not query_dir.is_absolute():
        query_dir = PROJECT_ROOT / query_dir

    processor, model = load_model(config)
    index, labels = load_index_and_labels(config)

    query_paths = sorted(query_dir.glob("*.jpg"))
    if not query_paths:
        raise FileNotFoundError(f"No .jpg files found in {query_dir}")

    correct = 0
    for query_path in tqdm(query_paths, desc="Evaluating"):
        query_id = get_person_id(query_path.name)
        matches = search_matches(query_path, processor, model, index, labels, top_k=1)
        top1_filename = matches[0][0]
        if top1_filename.startswith(query_id):
            correct += 1

    accuracy = correct / len(query_paths) * 100
    print(f"\nRank-1 Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{len(query_paths)}")


if __name__ == "__main__":
    config = load_config()
    evaluate(config)