"""Utilities for building LLCM annotations from an ORBench installation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_llcm_data_captions(
    orbench_root: str | Path,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build CSDT's LLCM annotation records from ORBench source annotations.

    The function keeps only LLCM samples. Each training visible image-text pair is
    paired with one randomly selected NIR image from the same original LLCM
    identity. Test RGB images receive a unique same-identity caption, while test
    NIR images and text queries are represented as separate records.
    """
    root = Path(orbench_root)
    train_annotations = _load_json(root / "train_annos.json")
    test_protocol = _load_json(root / "test_gallery_and_queries.json")
    random_generator = random.Random(seed)

    nir_by_identity = defaultdict(list)
    for nir_path in sorted((root / "nir").rglob("*llcm*")):
        if nir_path.is_file():
            relative_path = nir_path.relative_to(root).as_posix()
            nir_by_identity[_llcm_identity_key(relative_path)].append(relative_path)

    records = []
    for annotation in train_annotations:
        visible_path = annotation["file_path"]
        if not _is_llcm(visible_path):
            continue

        identity = annotation["id"]
        caption = annotation["caption"]
        records.append(
            {
                "id": identity,
                "file_path": visible_path,
                "caption": [caption],
                "split": "train",
                "type": "visible",
            }
        )

        nir_paths = nir_by_identity[_llcm_identity_key(visible_path)]
        if not nir_paths:
            raise ValueError(f"No matching NIR image for {visible_path}")
        records.append(
            {
                "id": identity,
                "file_path": random_generator.choice(nir_paths),
                "caption": [caption],
                "split": "train",
                "type": "infrared",
            }
        )

    llcm_test_ids = set()
    rgb_entries = []
    for identity, visible_path in test_protocol["RGB_GALLERY"]:
        if _is_llcm(visible_path):
            llcm_test_ids.add(identity)
            rgb_entries.append((identity, visible_path))

    text_pool = defaultdict(list)
    for identity, caption in test_protocol["TEXT"]:
        if identity in llcm_test_ids:
            text_pool[identity].append(caption)
    for captions in text_pool.values():
        random_generator.shuffle(captions)

    for identity, visible_path in rgb_entries:
        if not text_pool[identity]:
            raise ValueError(f"No remaining test caption for RGB image {visible_path}")
        records.append(
            {
                "id": identity,
                "file_path": visible_path,
                "caption": [text_pool[identity].pop()],
                "split": "test",
                "type": "visible",
            }
        )

    for identity, nir_path in test_protocol["NIR"]:
        if _is_llcm(nir_path):
            llcm_test_ids.add(identity)
            records.append(
                {
                    "id": identity,
                    "file_path": nir_path,
                    "caption": [],
                    "split": "test",
                    "type": "infrared",
                }
            )

    for identity, captions in text_pool.items():
        if captions:
            raise ValueError(f"Unused test captions for identity {identity}")

    for identity, caption in test_protocol["TEXT"]:
        if identity in llcm_test_ids:
            records.append(
                {
                    "id": identity,
                    "file_path": "",
                    "caption": [caption],
                    "split": "test",
                    "type": "infrared",
                }
            )

    if output_path is not None:
        destination = Path(output_path)
        with destination.open("w", encoding="utf-8") as output_file:
            json.dump(records, output_file, ensure_ascii=False)

    return records


def _llcm_identity_key(file_path: str) -> str:
    return "_".join(Path(file_path).name.split("_")[:3])


def _is_llcm(value: str) -> bool:
    return "llcm" in value.lower()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)
