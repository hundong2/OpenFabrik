"""Create a lightweight JSON quality report for a YOLO dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main() -> None:
    """Build and write a dataset quality report."""
    args = _parse_args()
    report = build_report(dataset=args.dataset)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(rendered)


def build_report(dataset: Path) -> dict[str, object]:
    """Summarize images, annotations, class balance, and empty labels.

    Args:
        dataset: Dataset root containing ``images`` and ``labels`` directories.

    Returns:
        A JSON-serializable report.

    Raises:
        NotADirectoryError: If the expected directories are missing.
    """
    images_root = dataset / "images"
    labels_root = dataset / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise NotADirectoryError(f"Expected {images_root} and {labels_root}")

    images = [path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    label_paths = list(labels_root.rglob("*.txt"))
    counts, per_image, empty_labels, malformed_rows = _read_labels(label_paths=label_paths)
    names = _read_class_names(dataset=dataset)
    class_counts = {_class_name(class_id=key, names=names): value for key, value in sorted(counts.items())}
    nonzero = list(counts.values())
    imbalance_ratio = max(nonzero) / min(nonzero) if nonzero else None

    return {
        "dataset": str(dataset.resolve()),
        "image_count": len(images),
        "label_file_count": len(label_paths),
        "annotation_count": sum(per_image),
        "empty_label_count": empty_labels,
        "malformed_row_count": malformed_rows,
        "annotations_per_labeled_image": {
            "mean": round(mean(per_image), 3) if per_image else 0.0,
            "median": median(per_image) if per_image else 0.0,
            "maximum": max(per_image, default=0),
        },
        "class_instance_counts": class_counts,
        "class_imbalance_ratio_max_over_min": round(imbalance_ratio, 3) if imbalance_ratio else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="YOLO dataset root")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def _read_labels(label_paths: list[Path]) -> tuple[Counter[int], list[int], int, int]:
    counts: Counter[int] = Counter()
    per_image: list[int] = []
    empty_labels = 0
    malformed_rows = 0
    for path in label_paths:
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            empty_labels += 1
        valid_in_file = 0
        for tokens in rows:
            try:
                class_id = int(tokens[0])
            except (IndexError, ValueError):
                malformed_rows += 1
                continue
            counts[class_id] += 1
            valid_in_file += 1
        per_image.append(valid_in_file)
    return counts, per_image, empty_labels, malformed_rows


def _read_class_names(dataset: Path) -> list[str]:
    path = dataset / "classes.txt"
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _class_name(class_id: int, names: list[str]) -> str:
    if 0 <= class_id < len(names):
        return f"{class_id}:{names[class_id]}"
    return str(class_id)


if __name__ == "__main__":
    main()
