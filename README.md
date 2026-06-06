# Person Re-Identification

Person Re-Identification (ReID) is the task of matching a person observed in one camera view against images captured by other cameras. This prototype implements a zero-shot ReID pipeline: it uses pretrained models only (no fine-tuning on Market-1501) to extract appearance embeddings from person crops, index a gallery with Faiss, and retrieve the top-K most similar gallery images for each query. The benchmark run uses the Market-1501 dataset (19,732 gallery images, 3,368 query images, 6 cameras) to measure retrieval accuracy under cross-camera conditions.

## Pipeline

```
Camera frames / crops
        |
        v
   [Detection]          YOLOv8 — detect persons, crop bounding boxes
        |
        v
   [Embedding]          DINOv2 (ViT-B/14) — 768-dim CLS token
        |
        v
   [Indexing]           Faiss IndexFlatIP — L2-normalized gallery vectors
        |
        v
   [Search]             Cosine similarity search, top-K retrieval
        |
        v
   [Evaluation]         Rank-1 / Recall@5 on Market-1501 queries
```

On Market-1501, gallery and query images are pre-cropped bounding boxes, so the detection step is skipped during benchmark evaluation. YOLOv8 and ByteTrack are included in the stack for the live-camera extension described below.

## Tech Stack

- **DINOv2 (ViT-B/14)** — feature extraction, 768-dim CLS token embeddings
- **YOLOv8** — person detection and bounding box cropping
- **Faiss (CPU)** — approximate nearest neighbor search
- **ByteTrack** — multi-object tracking across frames
- **Python 3.12, PyTorch, HuggingFace Transformers** — runtime and model loading

## Project Structure

```
person-reid/
├── config.yaml
├── data/Market_1501_dataset/
│   ├── bounding_box_test/   # gallery (19,732 images)
│   └── query/               # queries (3,368 images)
└── src/
    ├── extract.py    # DINOv2 feature extraction
    ├── index.py      # builds Faiss index from gallery embeddings
    ├── search.py     # queries the index for top-K matches
    └── eval.py       # computes Rank-1 / Recall@5 metrics
```

## Setup

Create a virtual environment and install dependencies from the project root:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

Download the [Market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501/data) dataset and place it under `data/`:

```
data/Market_1501_dataset/
├── bounding_box_test/
└── query/
```

Paths are configured in `config.yaml`. The `embeddings/` directory is created automatically when the index is built.

## How to Run

Run the scripts in order from the project root.

**1. Build the gallery index**

```bash
python src/index.py
```

Extracts DINOv2 embeddings for all gallery images and writes `embeddings/gallery.index` and `embeddings/labels.npy`. Expect a progress bar over 19,732 images; runtime depends on CPU throughput.

**2. Search a single query**

```bash
python src/search.py
```
we have the hardcoded image name just for the verification of the pipeline working before executing the eval.py

Runs retrieval for the default query image (`0001_c1s1_001051_00.jpg`) and prints the top-K matches with similarity scores:

the following is a example output
```
Rank 1: 0001_c1s1_001051_00.jpg, similarity: 0.9234
Rank 2: ...
```

**3. Evaluate on the full query set**

```bash
python src/eval.py
```

Iterates over all 3,368 query images, compares the top-1 retrieved person ID against the ground-truth ID encoded in the filename prefix, and prints aggregate accuracy:

```
Rank-1 Accuracy: XX.XX%
Correct: XXXX/3368
```

**4. Verify feature extraction (optional)**

```bash
python src/extract.py
```

Loads DINOv2, embeds a single query image, and prints the output shape:

```
(768,)
```

## Results

| Metric | Value |
|---|---|
| Rank-1 Accuracy | |
| Recall@5 | |

## Real-World Extension

In a live deployment, camera streams would be processed frame-by-frame: YOLOv8 detects persons and crops each bounding box, ByteTrack assigns a persistent track ID across frames within a camera, and DINOv2 embeds each crop on entry into the scene. New embeddings are compared against a Faiss index of known gallery identities (e.g., watchlist or previously enrolled subjects). When similarity exceeds the threshold configured in `config.yaml`, the system flags a match and associates it with the track ID for downstream alerting or logging. The Market-1501 benchmark isolates the embedding and retrieval stages; detection and tracking would be added as preprocessing before the same index-and-search core used here.
