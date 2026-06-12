from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from extract import PROJECT_ROOT, load_config
from search import load_index_and_labels, search_matches


SEED = 42
FIGURE_DPI = 150
MATCH_BORDER_WIDTH = 4
TERMINAL_BG = "#11161C"
BLUE_TINT = "#4C9AFF"
GREEN = "#2EAD4A"
RED = "#D14343"
QUERY_BORDER = "#333333"


@dataclass(frozen=True)
class QueryRecord:
    filename: str
    path: Path
    embedding: np.ndarray

    @property
    def person_id(self) -> str:
        return get_person_id(self.filename)

    @property
    def camera_id(self) -> str:
        return get_camera_id(self.filename)


@dataclass(frozen=True)
class MatchRecord:
    filename: str
    path: Path
    score: float

    @property
    def person_id(self) -> str:
        return get_person_id(self.filename)

    @property
    def camera_id(self) -> str:
        return get_camera_id(self.filename)


@dataclass(frozen=True)
class QueryResult:
    query: QueryRecord
    matches: list[MatchRecord]

    @property
    def top1(self) -> MatchRecord | None:
        return self.matches[0] if self.matches else None


def decode_filename(value: str | bytes | np.bytes_ | np.str_) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get_person_id(filename: str) -> str:
    return Path(filename).name[:4]


def get_camera_id(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) > 1 and len(parts[1]) >= 2:
        return parts[1][:2]
    base_name = Path(filename).name
    return base_name[5:7]


def load_query_cache(config: dict) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(config["paths"]["query_embeddings"])
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    embeddings = np.load(cache_dir / "embeddings.npy").astype(np.float32)
    labels = np.load(cache_dir / "labels.npy", allow_pickle=True)
    return embeddings, labels


def resolve_dir(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_image_path(base_dir: Path, filename: str) -> Path | None:
    candidate = base_dir / filename
    return candidate if candidate.exists() else None


def build_query_records(config: dict) -> list[QueryRecord]:
    query_dir = resolve_dir(config["paths"]["query"])
    embeddings, labels = load_query_cache(config)
    records: list[QueryRecord] = []

    for index, raw_label in enumerate(labels):
        filename = decode_filename(raw_label)
        image_path = resolve_image_path(query_dir, filename)
        if image_path is None:
            continue
        records.append(QueryRecord(filename=filename, path=image_path, embedding=embeddings[index]))

    if not records:
        raise FileNotFoundError(f"No query images found under {query_dir}")
    return records


def collect_valid_matches(
    query: QueryRecord,
    index,
    gallery_labels: np.ndarray,
    gallery_dir: Path,
    desired_k: int,
) -> list[MatchRecord]:
    max_results = len(gallery_labels)
    search_k = min(max(desired_k * 3, 10), max_results)
    seen: set[str] = set()
    valid_matches: list[MatchRecord] = []

    while len(valid_matches) < desired_k and search_k <= max_results:
        raw_matches = search_matches(
            query.path,
            None,
            None,
            index,
            gallery_labels,
            top_k=search_k,
            precomputed_embedding=query.embedding,
        )
        for raw_filename, score in raw_matches:
            filename = decode_filename(raw_filename)
            if filename in seen:
                continue
            seen.add(filename)
            image_path = resolve_image_path(gallery_dir, filename)
            if image_path is None:
                continue
            valid_matches.append(MatchRecord(filename=filename, path=image_path, score=float(score)))
            if len(valid_matches) == desired_k:
                break

        if len(valid_matches) >= desired_k or search_k == max_results:
            break
        search_k = min(search_k + desired_k * 5, max_results)

    return valid_matches


def render_image(ax, image_path: Path, title: str, border_color: str, overlay_color: str | None = None) -> None:
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor=border_color,
            linewidth=MATCH_BORDER_WIDTH,
        )
    )
    if overlay_color is not None:
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                facecolor=overlay_color,
                edgecolor="none",
                alpha=0.18,
            )
        )
    ax.set_title(title, fontsize=9, pad=8)


def blank_axis(ax, message: str = "No image") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#F3F4F6")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#666666")


def plot_retrieval_grid(results: list[QueryResult], output_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(results),
        ncols=6,
        figsize=(18, max(2.8 * len(results), 6)),
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    if len(results) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, result in enumerate(results):
        query = result.query
        render_image(
            axes[row_index, 0],
            query.path,
            f"Query\n{query.person_id} | {query.camera_id}",
            QUERY_BORDER,
        )
        for col_index in range(1, 6):
            axis = axes[row_index, col_index]
            match_index = col_index - 1
            if match_index >= len(result.matches):
                blank_axis(axis)
                continue
            match = result.matches[match_index]
            is_correct = match.person_id == query.person_id
            render_image(
                axis,
                match.path,
                f"{match.score:.2f}\n{match.person_id} | {match.camera_id}",
                GREEN if is_correct else RED,
            )

    fig.suptitle("Top-5 Retrieval Grid", fontsize=16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def classify_rank1_examples(pool: list[QueryResult]) -> tuple[list[QueryResult], list[QueryResult]]:
    correct: list[QueryResult] = []
    incorrect: list[QueryResult] = []

    for result in pool:
        top1 = result.top1
        if top1 is None:
            continue
        if top1.person_id == result.query.person_id:
            if len(correct) < 3:
                correct.append(result)
        else:
            if len(incorrect) < 3:
                incorrect.append(result)
        if len(correct) == 3 and len(incorrect) == 3:
            break

    return correct, incorrect


def plot_correct_vs_incorrect(
    correct_examples: list[QueryResult],
    incorrect_examples: list[QueryResult],
    output_path: Path,
) -> None:
    rows = 3
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=4,
        figsize=(16, 11),
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )

    for row_index in range(rows):
        if row_index < len(correct_examples):
            result = correct_examples[row_index]
            render_image(
                axes[row_index, 0],
                result.query.path,
                f"Correct Query\n{result.query.person_id} | {result.query.camera_id}",
                QUERY_BORDER,
            )
            top1 = result.top1
            render_image(
                axes[row_index, 1],
                top1.path,
                f"{top1.score:.2f}\n✓ CORRECT\n{top1.person_id} | {top1.camera_id}",
                GREEN,
            )
        else:
            blank_axis(axes[row_index, 0])
            blank_axis(axes[row_index, 1])

        if row_index < len(incorrect_examples):
            result = incorrect_examples[row_index]
            render_image(
                axes[row_index, 2],
                result.query.path,
                f"Incorrect Query\n{result.query.person_id} | {result.query.camera_id}",
                QUERY_BORDER,
            )
            top1 = result.top1
            render_image(
                axes[row_index, 3],
                top1.path,
                f"{top1.score:.2f}\n✗ INCORRECT\n{top1.person_id} | {top1.camera_id}",
                RED,
            )
        else:
            blank_axis(axes[row_index, 2])
            blank_axis(axes[row_index, 3])

    fig.suptitle("Correct vs Incorrect Rank-1 Examples", fontsize=16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def select_cross_camera_queries(ordered_records: list[QueryRecord]) -> list[QueryRecord]:
    by_camera: dict[str, list[QueryRecord]] = {}
    for record in ordered_records:
        by_camera.setdefault(record.camera_id, []).append(record)

    selected: list[QueryRecord] = []
    used_cameras: set[str] = set()
    preferred_order = ["c1", "c2", "c3", "c4", "c5", "c6"]

    for camera_id in preferred_order:
        candidates = by_camera.get(camera_id, [])
        if not candidates or camera_id in used_cameras:
            continue
        selected.append(candidates[0])
        used_cameras.add(camera_id)
        if len(selected) == 4:
            return selected

    for record in ordered_records:
        if record.camera_id in used_cameras:
            continue
        selected.append(record)
        used_cameras.add(record.camera_id)
        if len(selected) == 4:
            break

    return selected


def plot_cross_camera(results: list[QueryResult], output_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(results),
        ncols=6,
        figsize=(18, max(2.8 * len(results), 6)),
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    if len(results) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, result in enumerate(results):
        query = result.query
        render_image(
            axes[row_index, 0],
            query.path,
            f"Query Cam: {query.camera_id}\n{query.person_id} | {query.camera_id}",
            QUERY_BORDER,
        )

        for col_index in range(1, 6):
            axis = axes[row_index, col_index]
            match_index = col_index - 1
            if match_index >= len(result.matches):
                blank_axis(axis)
                continue

            match = result.matches[match_index]
            is_correct = match.person_id == query.person_id
            overlay = BLUE_TINT if match.camera_id != query.camera_id else None
            render_image(
                axis,
                match.path,
                f"{match.score:.2f}\n{match.person_id} | {match.camera_id}",
                GREEN if is_correct else RED,
                overlay_color=overlay,
            )

    fig.suptitle("Cross-Camera Retrieval", fontsize=16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_eval_capture() -> tuple[str, dict[str, str]]:
    eval_script = PROJECT_ROOT / "src" / "eval.py"
    completed = subprocess.run(
        [sys.executable, str(eval_script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    terminal_text = stdout or stderr or "No output captured from eval.py"
    if stderr and stderr not in terminal_text:
        terminal_text = f"{terminal_text}\n\n[stderr]\n{stderr}"

    metrics = {
        "Rank-1": extract_metric(stdout, "Rank-1"),
        "Rank-5": extract_metric(stdout, "Rank-5"),
        "mAP": extract_metric(stdout, "mAP"),
    }
    return terminal_text, metrics


def extract_metric(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    return f"{match.group(1)}%" if match else "N/A"


def plot_pipeline_output(terminal_text: str, metrics: dict[str, str], output_path: Path) -> None:
    lines = terminal_text.splitlines() or ["No output captured from eval.py"]
    max_line_length = max(len(line) for line in lines)
    fig_width = max(12, min(18, max_line_length * 0.12))
    fig_height = max(7, min(12, len(lines) * 0.28 + 3.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)
    fig.patch.set_facecolor(TERMINAL_BG)
    ax.set_facecolor(TERMINAL_BG)
    ax.axis("off")

    ax.text(
        0.03,
        0.95,
        terminal_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        color="white",
        linespacing=1.35,
    )

    metric_positions = [(0.18, 0.10), (0.50, 0.10), (0.82, 0.10)]
    for (label, value), (x_pos, y_pos) in zip(metrics.items(), metric_positions):
        ax.text(
            x_pos,
            y_pos,
            f"{label}\n{value}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1F6FEB", edgecolor="white", linewidth=1.5),
        )

    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_query_results(
    records: list[QueryRecord],
    index,
    gallery_labels: np.ndarray,
    gallery_dir: Path,
    desired_k: int = 5,
) -> list[QueryResult]:
    results: list[QueryResult] = []
    for record in records:
        matches = collect_valid_matches(record, index, gallery_labels, gallery_dir, desired_k=desired_k)
        results.append(QueryResult(query=record, matches=matches))
    return results


def main() -> None:
    config = load_config()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    gallery_dir = resolve_dir(config["paths"]["gallery"])
    index, gallery_labels = load_index_and_labels(config)
    query_records = build_query_records(config)

    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(len(query_records))
    ordered_records = [query_records[index] for index in shuffled_indices]

    retrieval_records = ordered_records[:10]
    retrieval_results = build_query_results(retrieval_records, index, gallery_labels, gallery_dir, desired_k=5)
    plot_retrieval_grid(retrieval_results, results_dir / "retrieval_grid.png")

    correct_examples: list[QueryResult] = []
    incorrect_examples: list[QueryResult] = []
    for pool_size in (10, 30, len(ordered_records)):
        section_pool = ordered_records[: min(pool_size, len(ordered_records))]
        section_results = build_query_results(section_pool, index, gallery_labels, gallery_dir, desired_k=5)
        correct_examples, incorrect_examples = classify_rank1_examples(section_results)
        if len(correct_examples) >= 3 and len(incorrect_examples) >= 3:
            break
    plot_correct_vs_incorrect(
        correct_examples[:3],
        incorrect_examples[:3],
        results_dir / "correct_vs_incorrect.png",
    )

    cross_camera_queries = select_cross_camera_queries(ordered_records)
    cross_camera_results = build_query_results(cross_camera_queries, index, gallery_labels, gallery_dir, desired_k=5)
    plot_cross_camera(cross_camera_results, results_dir / "cross_camera.png")

    terminal_text, metrics = run_eval_capture()
    plot_pipeline_output(terminal_text, metrics, results_dir / "pipeline_output.png")

    print("Saved: results/retrieval_grid.png")
    print("Saved: results/correct_vs_incorrect.png")
    print("Saved: results/cross_camera.png")
    print("Saved: results/pipeline_output.png")


if __name__ == "__main__":
    main()
