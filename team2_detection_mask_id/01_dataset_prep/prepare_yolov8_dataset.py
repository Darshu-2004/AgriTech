from __future__ import annotations

import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "drone_d_img"
SOURCE_IMAGES = SOURCE_ROOT / "images" / "train"
SOURCE_LABELS = SOURCE_ROOT / "labels" / "train"
DATASET_ROOT = ROOT / "dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def ensure_source_layout() -> None:
    if not SOURCE_IMAGES.exists():
        raise FileNotFoundError(f"Missing source images folder: {SOURCE_IMAGES}")
    if not SOURCE_LABELS.exists():
        raise FileNotFoundError(f"Missing source labels folder: {SOURCE_LABELS}")


def collect_labeled_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(SOURCE_IMAGES.glob("*")):
        if not image_path.is_file():
            continue
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    if not pairs:
        raise RuntimeError("No labeled image/label pairs were found.")
    return pairs


def build_split_map(pairs: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = pairs[:]
    random.Random(SEED).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    if train_count == 0 or val_count == 0 or test_count == 0:
        raise RuntimeError(
            f"Split produced an empty subset: train={train_count}, val={val_count}, test={test_count}"
        )

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def reset_dataset_root() -> None:
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    for split in ("train", "val", "test"):
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pairs(split_map: dict[str, list[tuple[Path, Path]]]) -> None:
    for split, items in split_map.items():
        for image_path, label_path in items:
            shutil.copy2(image_path, DATASET_ROOT / "images" / split / image_path.name)
            shutil.copy2(label_path, DATASET_ROOT / "labels" / split / label_path.name)


def write_data_yaml() -> None:
    content = "\n".join(
        [
            f"path: {DATASET_ROOT.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: pineapple",
            "",
        ]
    )
    (DATASET_ROOT / "data.yaml").write_text(content, encoding="utf-8")


def write_summary(split_map: dict[str, list[tuple[Path, Path]]], total_images: int) -> None:
    labeled_count = sum(len(items) for items in split_map.values())
    unlabeled_count = total_images - labeled_count
    lines = [
        "YOLOv8 dataset preparation summary",
        f"source_images: {total_images}",
        f"kept_labeled_pairs: {labeled_count}",
        f"removed_unlabeled_images: {unlabeled_count}",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"{split}: {len(split_map[split])}")
    lines.append("")
    (DATASET_ROOT / "split_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_source_layout()
    total_images = sum(1 for path in SOURCE_IMAGES.glob("*") if path.is_file())
    labeled_pairs = collect_labeled_pairs()
    split_map = build_split_map(labeled_pairs)
    reset_dataset_root()
    copy_pairs(split_map)
    write_data_yaml()
    write_summary(split_map, total_images)

    print(f"Prepared dataset at: {DATASET_ROOT}")
    print(f"Total source images: {total_images}")
    print(f"Labeled pairs kept: {len(labeled_pairs)}")
    print(f"Unlabeled images excluded: {total_images - len(labeled_pairs)}")
    for split in ("train", "val", "test"):
        print(f"{split}: {len(split_map[split])}")


if __name__ == "__main__":
    main()
