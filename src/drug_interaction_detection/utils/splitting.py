from __future__ import annotations

import hashlib


def deterministic_split(key: str, train_fraction: float, validation_fraction: float) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + validation_fraction:
        return "validation"
    return "test"
