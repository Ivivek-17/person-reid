# Person Re-Identification with DINOv2 and FAISS

This repository implements a compact person re-identification (ReID) pipeline for the Market-1501 dataset. It uses a pretrained DINOv2 vision transformer to extract image embeddings, stores gallery embeddings in a FAISS index, and performs cosine-similarity retrieval for query images.

The project is intentionally simple and modular:

- `extract.py` loads the model and produces a single image embedding.
- `index.py` builds a gallery index from gallery images.
- `search.py` retrieves the top-K nearest gallery matches for a query image.
- `eval.py` evaluates retrieval quality across the full query set.
- `visualize.py` generates visual review assets for supervisor or stakeholder inspection.

## Overview

Person ReID answers a practical question:

> Given a cropped person image from one camera, which gallery images most likely show the same person?

In this codebase, the flow is:

1. Load a pretrained image encoder.
2. Convert each gallery image into a fixed-length embedding vector.
3. L2-normalize embeddings and index them with FAISS.
4. Embed a query image or reuse a cached query embedding.
5. Search the gallery index using inner product on normalized vectors, which is equivalent to cosine similarity.
6. Report ranked matches and aggregate retrieval metrics.

## Current Capabilities

- Pretrained DINOv2 feature extraction with no task-specific fine-tuning.
- FAISS-based nearest-neighbor search over gallery embeddings.
- Optional reuse of precomputed query embeddings from `embeddings/query_cache/`.
- Top-K search results with similarity scores.
- Dataset-wide evaluation with Rank-1, Rank-5, and mean Average Precision (mAP).
- Visual reporting for qualitative review in `results/`.

## Repository Structure

```text
person-reid/
|-- config.yaml
|-- data/
|   `-- Market_1501_dataset/
|       |-- bounding_box_test/      # gallery images
|       `-- query/                  # query images
|-- embeddings/
|   |-- gallery.index               # FAISS gallery index
|   |-- labels.npy                  # gallery filenames aligned with the index
|   `-- query_cache/
|       |-- embeddings.npy          # cached query embeddings
|       `-- labels.npy              # cached query filenames
|-- results/                        # generated visual artifacts
|-- src/
|   |-- extract.py
|   |-- index.py
|   |-- search.py
|   |-- eval.py
|   `-- visualize.py
`-- tests/
    |-- test_cmc_fix.py
    `-- test_eval_protocol.py
```

## Technical Design

### Embedding Model

- Model: `facebook/dinov2-base`
- Embedding dimension: `768`
- Frameworks: PyTorch + Hugging Face Transformers

`src/extract.py` uses the CLS token from the last hidden state as the image representation.

### Retrieval Backend

- Backend: `faiss.IndexFlatIP`
- Similarity: inner product on L2-normalized vectors
- Effectively equivalent to cosine similarity after normalization

`src/index.py` normalizes gallery embeddings before indexing, and `src/search.py` normalizes each query embedding before search.

### Query Embedding Reuse

If `embeddings/query_cache/embeddings.npy` and `embeddings/query_cache/labels.npy` exist, `src/eval.py` and `src/visualize.py` reuse those cached query embeddings instead of recomputing them. This reduces evaluation time and avoids unnecessary inference.

## Dataset Assumptions

The code expects the Market-1501 naming convention:

```text
PPPP_CcSs_xxxxxx_xx.jpg
```

Example:

```text
0001_c1s1_001051_00.jpg
```

Where:

- `PPPP` is the person ID.
- `c1` is the camera ID.
- Additional segments identify sequence/frame metadata.

The current implementation derives the person identity from the filename prefix and uses camera metadata for visualization labels.

## Configuration

Project configuration lives in config.yaml

Example:

```yaml
model:
  name: "facebook/dinov2-base"
  embedding_dim: 768

reid:
  similarity_threshold: 0.75
  top_k: 5

paths:
  gallery: data/Market_1501_dataset/bounding_box_test
  query: data/Market_1501_dataset/query
  index: embeddings/gallery.index
  query_embeddings: embeddings/query_cache
```

All runtime paths are resolved relative to `PROJECT_ROOT` in `src/extract.py`.

## Environment Setup

### Prerequisites

- Python 3.12 recommended
- CPU-compatible PyTorch environment
- Market-1501 dataset placed under `data/`

### Install

From the project root:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you are using a different virtual environment name, activate that environment before running the scripts.

## Data Layout

Expected dataset layout:

```text
data/
`-- Market_1501_dataset/
    |-- bounding_box_test/
    `-- query/
```

Expected embedding artifacts:

```text
embeddings/
|-- gallery.index
|-- labels.npy
`-- query_cache/
    |-- embeddings.npy
    `-- labels.npy
```

## How to Run

Run all commands from the repository root.

### 1. Verify model loading and embedding extraction

```bash
python src/extract.py
```

Expected output:

```text
(768,)
```

This is a quick sanity check that the model loads and produces the configured embedding dimension.

### 2. Build the gallery index

```bash
python src/index.py
```

This script:

- reads every `.jpg` file in `config["paths"]["gallery"]`
- extracts a DINOv2 embedding per image
- L2-normalizes the full gallery matrix
- writes the FAISS index to `embeddings/gallery.index`
- writes aligned gallery labels to `embeddings/labels.npy`

### 3. Search a single query image

```bash
python src/search.py
```

The script currently uses a hardcoded query image for smoke testing:

```text
data/Market_1501_dataset/query/0001_c1s1_001051_00.jpg
```

Expected output format:

```text
Rank 1: 0001_c1s1_001051_00.jpg, similarity: 0.9234
Rank 2: ...
```

### 4. Evaluate retrieval performance

```bash
python src/eval.py
```

This script:

- iterates through all query `.jpg` files
- reuses cached query embeddings when available
- searches the gallery index
- computes Rank-1, Rank-5, and mAP
- prints aggregate metrics to stdout

Expected output format:

```text
Evaluation Metrics
------------------
Rank-1: 81.77% (correct/total)
Rank-5: 88.48% (correct/total)
mAP:    78.62%
```

### 5. Generate qualitative review visuals

```bash
python src/visualize.py
```

This script generates the following files under `results/`:

- `results/retrieval_grid.png`
- `results/correct_vs_incorrect.png`
- `results/cross_camera.png`
- `results/pipeline_output.png`

These assets are useful for supervisor review, demos, and fast qualitative validation of the retrieval pipeline.

## Evaluation Logic

The current `src/eval.py` logic works as follows:

- Query filenames are matched to person IDs using the filename prefix.
- If cached query embeddings are available, they are used directly.
- Search is executed with `top_k=10`.
- Rank-1 is counted as correct when the top match starts with the same person ID.
- Rank-5 is counted as correct when any of the first five matches share the same person ID.
- mAP is computed over the retrieved result list returned by the search step.

This is a practical retrieval benchmark for the current repository implementation.

## Visual Reporting

`src/visualize.py` is a standalone reporting utility that:

- samples queries reproducibly with random seed `42`
- reuses precomputed query embeddings from `embeddings/query_cache/`
- renders top-5 retrieval grids with score overlays
- distinguishes correct and incorrect matches visually
- highlights cross-camera retrieval behavior
- runs `eval.py` and converts terminal output into a presentation-friendly panel

This script is designed for review workflows where quantitative metrics alone are not sufficient.

## Outputs

### Index Artifacts

- `embeddings/gallery.index`
- `embeddings/labels.npy`

### Optional Query Cache

- `embeddings/query_cache/embeddings.npy`
- `embeddings/query_cache/labels.npy`

### Visualization Outputs

- `results/retrieval_grid.png`
- `results/correct_vs_incorrect.png`
- `results/cross_camera.png`
- `results/pipeline_output.png`

## Operational Notes

- The code resolves relative paths via `PROJECT_ROOT`, so commands should be run from a normal project checkout rather than copied into unrelated working directories.
- `search.py` and `eval.py` assume the FAISS gallery index and gallery labels already exist.
- `eval.py` benefits significantly from `embeddings/query_cache/` if the cache has already been created.
- Gallery and query images must remain aligned with the cached label files.
- `visualize.py` skips missing gallery image files rather than failing the entire report.

## Known Limitations

- The search entry point in `src/search.py` currently uses a hardcoded query path for smoke testing rather than a CLI argument.
- The repository is CPU-oriented by default and may be slow for full indexing on modest hardware.
- Retrieval quality depends entirely on the pretrained embedding model and index quality; there is no fine-tuning, reranking, or metric learning stage yet.
- There is no packaged command-line interface, experiment tracker, or model registry in the current version.

## Recommended Next Improvements

- Add CLI arguments for `search.py`, `eval.py`, and `visualize.py`.
- Add structured logging and run metadata capture.
- Add benchmark snapshots to the `Results` section once metrics are finalized.
- Add explicit protocol validation tests for dataset-specific evaluation rules.
- Add GPU support and batching for faster embedding extraction.

## Troubleshooting

### `No .jpg files found`

Check the gallery and query paths in config.yam and confirm the Market-1501 dataset is placed under `data/Market_1501_dataset/`.

### `gallery.index` or `labels.npy` missing

Run:

```bash
python src/index.py
```

### Slow evaluation

Make sure query embeddings exist in:

```text
embeddings/query_cache/
```

When present, `eval.py` and `visualize.py` reuse them instead of recomputing each query embedding.

## Summary

This project provides a clean baseline for zero-shot person re-identification using DINOv2 and FAISS. It is well-suited for:

- coursework and academic demos
- rapid prototyping of image-retrieval systems
- qualitative and quantitative ReID benchmarking
- extension into more advanced cross-camera tracking or deployment workflows

For the best experience, keep the index artifacts, cached query embeddings, dataset paths, and configuration file in sync before running evaluation or visualization.
