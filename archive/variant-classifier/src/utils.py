"""Shared helpers: logging, HGVS id construction, and nested-dict extraction."""
import logging
import os
from typing import Any, Optional, Sequence

_PURINES = {"A", "G"}
_PYRIMIDINES = {"C", "T"}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_hgvs(chrom: str, pos: str, ref: str, alt: str) -> Optional[str]:
    """Build an hg38 HGVS id understood by myvariant.info, e.g. 'chr1:g.35A>G'."""
    if not (chrom and pos and ref and alt):
        return None
    chrom = str(chrom).replace("chr", "")
    return f"chr{chrom}:g.{pos}{ref}>{alt}"


def is_transition(ref: str, alt: str) -> Optional[int]:
    """A<->G or C<->T is a transition (1); otherwise a transversion (0)."""
    if not ref or not alt or len(ref) != 1 or len(alt) != 1:
        return None
    if (ref in _PURINES and alt in _PURINES) or (
        ref in _PYRIMIDINES and alt in _PYRIMIDINES
    ):
        return 1
    return 0


def dig(obj: Any, path: str) -> Any:
    """Safely walk a dotted path through nested dicts/lists from the API.

    Lists are traversed element-wise so 'a.b' on [{'b':1},{'b':2}] -> [1,2].
    Returns None when any step is missing.
    """
    keys = path.split(".")
    cur: Any = obj
    for key in keys:
        if cur is None:
            return None
        if isinstance(cur, list):
            cur = [c.get(key) if isinstance(c, dict) else None for c in cur]
            cur = [c for c in cur if c is not None]
            cur = cur or None
        elif isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur


def first_numeric(value: Any, agg: str = "first") -> Optional[float]:
    """Collapse a value (possibly a list) into a single float via `agg`."""
    if value is None:
        return None
    if isinstance(value, list):
        nums = []
        for v in value:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                continue
        if not nums:
            return None
        if agg == "min":
            return min(nums)
        if agg == "max":
            return max(nums)
        if agg == "mean":
            return sum(nums) / len(nums)
        return nums[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
