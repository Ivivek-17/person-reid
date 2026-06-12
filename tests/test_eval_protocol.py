"""Unit tests for Market-1501 evaluation protocol helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval import (
    count_good_gallery_images,
    filter_ranked_results,
    get_camera_id,
    get_person_id,
    is_distractor,
    is_junk,
    parse_person_id,
)


def test_filename_parsing():
    name = "0001_c1s1_001051_00.jpg"
    assert get_person_id(name) == "0001"
    assert get_camera_id(name) == "c1"


def test_junk_same_camera_not_same_sequence():
    query = "0001_c1s1_001051_00.jpg"
    same_cam_diff_seq = "0001_c1s2_001200_00.jpg"
    cross_cam = "0001_c2s1_001051_00.jpg"
    other_person = "0042_c1s1_001051_00.jpg"

    assert is_junk(query, same_cam_diff_seq)
    assert not is_junk(query, cross_cam)
    assert not is_junk(query, other_person)


def test_distractor_detection():
    assert not is_distractor("1501_c1s1_001051_00.jpg")
    assert is_distractor("1502_c1s1_001051_00.jpg")
    assert is_distractor("-1_c1s1_001051_00.jpg")


def test_person_id_not_first_four_chars():
    assert get_person_id("-1_c1s1_001051_00.jpg") == "-1"
    assert parse_person_id("-1_c1s1_001051_00.jpg") == -1


def test_filter_ranked_results():
    query = "0001_c1s1_001051_00.jpg"
    gallery_labels = np.array(
        [
            "0001_c1s1_001051_00.jpg",  # junk — same camera
            "0001_c1s2_001200_00.jpg",  # junk — same camera, different sequence
            "1502_c2s1_001051_00.jpg",  # distractor
            "0001_c2s1_001051_00.jpg",  # good
            "0042_c3s1_001051_00.jpg",  # wrong person
        ]
    )
    indices = np.array([0, 1, 2, 3, 4])

    filtered = filter_ranked_results(query, indices, gallery_labels)
    assert filtered == ["0001_c2s1_001051_00.jpg", "0042_c3s1_001051_00.jpg"]


def test_count_good_gallery_images():
    query = "0001_c1s1_001051_00.jpg"
    gallery_labels = np.array(
        [
            "0001_c1s1_001051_00.jpg",
            "0001_c1s2_001200_00.jpg",
            "0001_c2s1_001051_00.jpg",
            "0001_c3s1_001051_00.jpg",
            "1502_c2s1_001051_00.jpg",
        ]
    )
    assert count_good_gallery_images(query, gallery_labels) == 2


def test_shallow_search_would_fail_without_full_gallery():
    """Simulate old bug: top-3 are junk, correct match is at index 3."""
    query = "0001_c1s1_001051_00.jpg"
    gallery_labels = np.array(
        [
            "0001_c1s1_001051_00.jpg",
            "0001_c1s2_001200_00.jpg",
            "0001_c1s3_001300_00.jpg",
            "0001_c2s1_001051_00.jpg",
        ]
    )
    indices = np.array([0, 1, 2, 3])

    shallow = filter_ranked_results(query, indices[:3], gallery_labels)
    full = filter_ranked_results(query, indices, gallery_labels)

    assert shallow == []
    assert full[0] == "0001_c2s1_001051_00.jpg"
