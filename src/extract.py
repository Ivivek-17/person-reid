from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config: dict) -> tuple[AutoImageProcessor, AutoModel]:
    model_name = config["model"]["name"]
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def get_embedding(
    image_path: str | Path,
    processor: AutoImageProcessor,
    model: AutoModel,
) -> np.ndarray:
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    cls_token = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return cls_token.numpy()


if __name__ == "__main__":
    config = load_config()
    processor, model = load_model(config)

    image_path = "data/Market_1501_dataset/query/0001_c1s1_001051_00.jpg"
    embedding = get_embedding(image_path, processor, model)
    print(embedding.shape)
