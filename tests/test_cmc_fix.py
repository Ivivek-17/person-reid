"""Verify the CMC cumulative fix in compute_rank_metrics."""
import sys
from pathlib import Path

import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval import compute_rank_metrics
from test_eval_protocol import *

# Run all existing tests first
test_filename_parsing()
test_junk_same_camera_not_same_sequence()
test_distractor_detection()
test_person_id_not_first_four_chars()
test_filter_ranked_results()
test_count_good_gallery_images()
test_shallow_search_would_fail_without_full_gallery()
print("All existing tests PASSED")


# Verify CMC cumulative property with controlled data
dim = 4
gallery_embs = np.eye(dim + 1, dim, dtype=np.float32)  # 5 orthogonal-ish vectors
faiss.normalize_L2(gallery_embs)
index = faiss.IndexFlatIP(dim)
index.add(gallery_embs)

gallery_labels = np.array([
    "0001_c1s1_001_00.jpg",  # junk (same cam as query)
    "0042_c2s1_001_00.jpg",  # wrong person
    "0001_c2s1_001_00.jpg",  # correct (cross-cam)
    "0042_c3s1_001_00.jpg",  # wrong person
    "0042_c4s1_001_00.jpg",  # wrong person
])

# Use the correct-match embedding as the query
query_emb = gallery_embs[2:3].copy()
metrics = compute_rank_metrics(
    query_emb, ["0001_c1s1_001_00.jpg"], index, gallery_labels, max_rank=5
)
r1, r5 = metrics["rank1"], metrics["rank5"]
print(f"rank1={r1}, rank5={r5}")
assert r5 >= r1, f"FAIL: rank5 ({r5}) < rank1 ({r1}) — CMC violated!"
print("CMC cumulative property verified: rank5 >= rank1")
print("ALL TESTS PASSED")
