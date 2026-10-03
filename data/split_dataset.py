"""Create a deterministic sample-level split for the tri-modal training set."""

import os
import random
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".pfm", ".tif", ".tiff", ".webp"}


def _resolve_modality(root, name):
    names = ("infrared", "infared") if name in {"infrared", "infared"} else (name,)
    for candidate in names:
        directory = root / candidate
        if directory.is_dir():
            return directory
    return None


def _image_files(directory):
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(directory).as_posix(),
    )


def _validate_existing_split(split_root, expected_relative_paths, expected_labels):
    if not split_root.exists():
        return
    for modality in ("visible", "infrared", "infared", "depth"):
        modality_root = split_root / modality
        if modality_root.is_dir():
            actual = {path.relative_to(modality_root) for path in _image_files(modality_root)}
            if actual - expected_relative_paths:
                raise RuntimeError(
                    f"Existing split directory {modality_root} contains samples outside the deterministic split. "
                    "Move the existing train/val directories aside before creating the split."
                )
    labels_root = split_root / "labels"
    if labels_root.is_dir():
        actual_labels = {path.name for path in labels_root.glob("*.txt")}
        if actual_labels - expected_labels:
            raise RuntimeError(
                f"Existing split directory {labels_root} contains labels outside the deterministic split. "
                "Move the existing train/val directories aside before creating the split."
            )


def _verify_created_splits(root, modality_dirs, train_paths, val_paths, labels_dir):
    if train_paths & val_paths:
        raise RuntimeError("Generated train and validation sample sets overlap.")

    split_samples = {}
    for split_name, expected_paths in (("train", train_paths), ("val", val_paths)):
        split_root = root / split_name
        for source_root in modality_dirs.values():
            modality_root = split_root / source_root.name
            actual_paths = {path.relative_to(modality_root) for path in _image_files(modality_root)}
            if actual_paths != expected_paths:
                missing = sorted(expected_paths - actual_paths)
                extra = sorted(actual_paths - expected_paths)
                raise RuntimeError(
                    f"Invalid {split_name} split in {modality_root}: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
        split_samples[split_name] = {
            path.relative_to(split_root / next(iter(modality_dirs.values())).name)
            for path in _image_files(split_root / next(iter(modality_dirs.values())).name)
        }

        expected_label_names = {
            f"{relative_path.stem}.txt"
            for relative_path in expected_paths
            if (labels_dir / f"{relative_path.stem}.txt").is_file()
        }
        actual_label_names = {path.name for path in (split_root / "labels").glob("*.txt")}
        if actual_label_names != expected_label_names:
            raise RuntimeError(
                f"Invalid labels in {split_root / 'labels'}: "
                f"missing={sorted(expected_label_names - actual_label_names)[:5]}, "
                f"extra={sorted(actual_label_names - expected_label_names)[:5]}"
            )

    overlap = split_samples["train"] & split_samples["val"]
    if overlap:
        raise RuntimeError(f"Train/validation leakage detected: {sorted(overlap)[:5]}")
    if split_samples["train"] | split_samples["val"] != train_paths | val_paths:
        raise RuntimeError("The generated train and validation splits do not cover all source samples.")


def _copy_sample(source_root, destination_root, relative_path):
    source = source_root / relative_path
    destination = destination_root / relative_path
    if not source.is_file():
        raise FileNotFoundError(f"Missing expected paired file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    destination_stat = destination.stat() if destination.exists() else None
    if destination_stat and (
        destination_stat.st_size == source_stat.st_size
        and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        and cv2.imread(str(destination)) is not None
    ):
        return False
    if cv2.imread(str(source)) is None:
        raise RuntimeError(f"Source image cannot be decoded by OpenCV: {source}")

    temporary = destination.with_name(f"{destination.name}.split-tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def split_dataset(root, val_ratio=0.15, seed=42):
    """Copy paired RGB/IR/Depth/label samples into stable train and val directories."""
    root = Path(root).resolve()
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    modality_dirs = {
        "visible": _resolve_modality(root, "visible"),
        "infrared": _resolve_modality(root, "infrared"),
        "depth": _resolve_modality(root, "depth"),
    }
    missing = [name for name, directory in modality_dirs.items() if directory is None]
    labels_dir = root / "labels"
    if missing or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected visible/, infrared/ (or infared/), depth/, and labels/ under dataset root {root}; "
            f"missing modalities: {missing}"
        )

    file_maps = {
        name: {path.relative_to(directory): path for path in _image_files(directory)}
        for name, directory in modality_dirs.items()
    }
    visible_paths = set(file_maps["visible"])
    if not visible_paths:
        raise FileNotFoundError(f"No supported images found under {modality_dirs['visible']}")
    for name in ("infrared", "depth"):
        if set(file_maps[name]) != visible_paths:
            missing_paths = sorted(visible_paths - set(file_maps[name]))
            extra_paths = sorted(set(file_maps[name]) - visible_paths)
            raise ValueError(
                f"Tri-modal pairing mismatch in {modality_dirs[name]}: "
                f"missing={missing_paths[:5]}, extra={extra_paths[:5]}"
            )

    stems = [relative_path.stem for relative_path in visible_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Image stems must be unique because YOLO label paths are based on image stems.")
    if len(stems) < 2:
        raise ValueError("At least two images are required to create separate train and validation sets.")

    sample_stems = set(stems)
    extra_labels = {path.stem for path in labels_dir.glob("*.txt")} - sample_stems
    if extra_labels:
        raise ValueError(f"Labels without matching images found under {labels_dir}: {sorted(extra_labels)[:5]}")

    ordered_paths = sorted(visible_paths, key=lambda path: path.as_posix())
    shuffled_paths = ordered_paths.copy()
    random.Random(seed).shuffle(shuffled_paths)
    val_count = min(len(shuffled_paths) - 1, max(1, round(len(shuffled_paths) * val_ratio)))
    val_paths = set(shuffled_paths[:val_count])
    train_paths = set(shuffled_paths[val_count:])
    if train_paths & val_paths:
        raise RuntimeError("Train and validation split unexpectedly contain overlapping samples.")

    refreshed_files = 0
    for split_name, relative_paths in (("train", train_paths), ("val", val_paths)):
        split_root = root / split_name
        expected_labels = {
            f"{relative_path.stem}.txt"
            for relative_path in relative_paths
            if (labels_dir / f"{relative_path.stem}.txt").is_file()
        }
        _validate_existing_split(split_root, relative_paths, expected_labels)
        for source_root in modality_dirs.values():
            destination_root = split_root / source_root.name
            for relative_path in relative_paths:
                refreshed_files += _copy_sample(source_root, destination_root, relative_path)
        destination_labels = split_root / "labels"
        destination_labels.mkdir(parents=True, exist_ok=True)
        for relative_path in relative_paths:
            label_name = f"{relative_path.stem}.txt"
            source = labels_dir / label_name
            if source.is_file():
                destination = destination_labels / label_name
                source_stat = source.stat()
                destination_stat = destination.stat() if destination.exists() else None
                if not destination_stat or (
                    destination_stat.st_size != source_stat.st_size
                    or destination_stat.st_mtime_ns != source_stat.st_mtime_ns
                ):
                    temporary = destination.with_name(f"{destination.name}.split-tmp")
                    try:
                        shutil.copy2(source, temporary)
                        os.replace(temporary, destination)
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                    refreshed_files += 1

    if refreshed_files:
        print(f"Refreshed {refreshed_files} missing or stale split files from the source dataset.")

    _verify_created_splits(root, modality_dirs, train_paths, val_paths, labels_dir)
    print(
        f"Verified split: train={len(train_paths)}, val={len(val_paths)}, overlap=0, "
        f"modalities={', '.join(source_root.name for source_root in modality_dirs.values())}"
    )
    return root / "train", root / "val", len(train_paths), len(val_paths)
