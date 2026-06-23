"""Step 2 — Annotate variants with functional scores via myvariant.info.

For each variant we pull a compact set of well-established predictors and
population data (CADD, conservation, SIFT, PolyPhen, REVEL, gnomAD AF) using
the myvariant.info REST API. No giant local databases required.

Note: CADD is pulled deliberately so we can use it as a *baseline* predictor
to benchmark our trained model against in `evaluate.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import (LABELED_CSV, ANNOTATIONS_CSV, DATA_DIR, ANNOTATION_BATCH)
from utils import get_logger, ensure_dir, dig, first_numeric

log = get_logger("annotate")

# (output_column, dotted_api_path, list_aggregator)
FIELD_MAP = [
    ("cadd_phred", "cadd.phred", "max"),
    ("gerp", "cadd.gerp.rs", "max"),
    ("sift_score", "dbnsfp.sift.score", "min"),      # lower = more damaging
    ("polyphen_score", "dbnsfp.polyphen2.hdiv.score", "max"),
    ("revel_score", "dbnsfp.revel.score", "max"),
    ("gnomad_af", "gnomad_genome.af.af", "max"),
]
API_FIELDS = [path for _, path, _ in FIELD_MAP]


def annotate(ids):
    import myvariant
    mv = myvariant.MyVariantInfo()
    rows = []
    n_batches = (len(ids) + ANNOTATION_BATCH - 1) // ANNOTATION_BATCH
    for b in range(n_batches):
        batch = ids[b * ANNOTATION_BATCH:(b + 1) * ANNOTATION_BATCH]
        log.info("Annotating batch %d/%d (%d variants)", b + 1, n_batches,
                 len(batch))
        res = mv.getvariants(batch, fields=API_FIELDS, assembly="hg38")
        for item in res:
            row = {"variant_id": item.get("query")}
            for col, path, agg in FIELD_MAP:
                row[col] = first_numeric(dig(item, path), agg)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dir(DATA_DIR)
    labeled = pd.read_csv(LABELED_CSV)
    ids = labeled["variant_id"].dropna().unique().tolist()
    log.info("Annotating %d unique variants ...", len(ids))
    ann = annotate(ids)
    # one row per variant_id (API can echo duplicates)
    ann = ann.drop_duplicates("variant_id")
    coverage = {c: f"{ann[c].notna().mean():.0%}"
                for c, _, _ in FIELD_MAP}
    log.info("Annotation coverage: %s", coverage)
    ann.to_csv(ANNOTATIONS_CSV, index=False)
    log.info("Wrote -> %s", ANNOTATIONS_CSV)


if __name__ == "__main__":
    main()
