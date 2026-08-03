"""Validate a YOLO detection or segmentation dataset without GPU dependencies."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main() -> int:
    """Validate the requested dataset and return a process exit code."""
    args = _parse_args()
    errors, warnings = validate_dataset(dataset=args.dataset)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Validation complete: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


def validate_dataset(dataset: Path) -> tuple[list[str], list[str]]:
    """Validate image-label pairing, YOLO rows, and exact split leakage.

    Args:
        dataset: Dataset root containing ``images`` and ``labels`` directories.

    Returns:
        A pair containing errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    images_root = dataset / "images"
    labels_root = dataset / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        return [f"Expected {images_root} and {labels_root} directories"], warnings

    images = _collect_images(images_root=images_root)
    labels = list(labels_root.rglob("*.txt"))
    image_keys = {_relative_key(path=path, root=images_root): path for path in images}
    label_keys = {_relative_key(path=path, root=labels_root): path for path in labels}

    for key in sorted(image_keys.keys() - label_keys.keys()):
        warnings.append(f"Image has no label file: {image_keys[key]}")
    for key in sorted(label_keys.keys() - image_keys.keys()):
        errors.append(f"Label has no matching image: {label_keys[key]}")

    class_count = _read_class_count(dataset=dataset)
    for label_path in labels:
        errors.extend(_validate_label_file(path=label_path, class_count=class_count))
    errors.extend(_find_exact_split_leakage(images=images, images_root=images_root))
    if not images:
        errors.append("No supported image files found")
    return errors, warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="YOLO dataset root")
    return parser.parse_args()


def _collect_images(images_root: Path) -> list[Path]:
    return [path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]


def _relative_key(path: Path, root: Path) -> Path:
    return path.relative_to(root).with_suffix("")


def _read_class_count(dataset: Path) -> int | None:
    classes_file = dataset / "classes.txt"
    if not classes_file.is_file():
        return None
    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(names)


def _validate_label_file(path: Path, class_count: int | None) -> list[str]:
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        error = _validate_label_row(row=line, class_count=class_count)
        if error:
            errors.append(f"{path}:{line_number}: {error}")
    return errors


def _validate_label_row(row: str, class_count: int | None) -> str | None:
    tokens = row.split()
    if len(tokens) != 5 and (len(tokens) < 7 or len(tokens) % 2 == 0):
        return "Expected 5 detection tokens or class_id plus at least 3 polygon points"
    try:
        class_id = int(tokens[0])
        coordinates = [float(token) for token in tokens[1:]]
    except ValueError:
        return "Class ID must be an integer and coordinates must be numeric"
    if class_id < 0 or (class_count is not None and class_id >= class_count):
        return f"Class ID {class_id} is outside the configured range"
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        return "Coordinates must be normalized to [0, 1]"
    if len(tokens) == 5 and (coordinates[2] <= 0.0 or coordinates[3] <= 0.0):
        return "Bounding-box width and height must be positive"
    return None


def _find_exact_split_leakage(images: list[Path], images_root: Path) -> list[str]:
    hashes: dict[str, tuple[str, Path]] = {}
    errors: list[str] = []
    for path in images:
        relative = path.relative_to(images_root)
        split = relative.parts[0] if len(relative.parts) > 1 else "flat"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = hashes.get(digest)
        if previous and previous[0] != split:
            errors.append(f"Exact duplicate crosses splits: {previous[1]} <-> {path}")
        else:
            hashes[digest] = (split, path)
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
