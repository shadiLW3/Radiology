"""Step 1 — Download ClinVar and build a labeled set of SNVs.

Streams the ClinVar `variant_summary.txt.gz` release, keeps high-confidence
single-nucleotide variants on GRCh38 with an unambiguous benign/pathogenic
label, deduplicates, stratified-samples down to MAX_VARIANTS, and writes
`data/clinvar_labeled.csv`.
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import (CLINVAR_URL, RAW_CLINVAR_GZ, LABELED_CSV, DATA_DIR,
                    RANDOM_SEED, MAX_VARIANTS, MIN_REVIEW_STARS)
from utils import get_logger, ensure_dir, build_hgvs, is_transition

log = get_logger("download")

PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}

# ClinVar ReviewStatus -> star confidence (abbreviated from ClinVar docs).
REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, conflicting classifications": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
}

USECOLS = [
    "Type", "ClinicalSignificance", "GeneSymbol", "Assembly", "Chromosome",
    "ReviewStatus", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF",
]


def download() -> None:
    ensure_dir(DATA_DIR)
    if os.path.exists(RAW_CLINVAR_GZ):
        log.info("ClinVar archive already present: %s", RAW_CLINVAR_GZ)
        return
    log.info("Downloading ClinVar (~80 MB) from %s ...", CLINVAR_URL)
    urllib.request.urlretrieve(CLINVAR_URL, RAW_CLINVAR_GZ)
    log.info("Saved -> %s", RAW_CLINVAR_GZ)


def label_significance(sig: str) -> int:
    sig = (sig or "").strip()
    if sig in PATHOGENIC:
        return 1
    if sig in BENIGN:
        return 0
    return -1


def load_and_filter() -> pd.DataFrame:
    log.info("Reading + filtering ClinVar in chunks ...")
    kept = []
    reader = pd.read_csv(
        RAW_CLINVAR_GZ, sep="\t", chunksize=200_000, dtype=str,
        usecols=USECOLS, low_memory=False, compression="gzip",
    )
    for i, chunk in enumerate(reader):
        chunk = chunk[chunk["Assembly"] == "GRCh38"]
        chunk = chunk[chunk["Type"] == "single nucleotide variant"]
        chunk = chunk.assign(
            label=chunk["ClinicalSignificance"].map(label_significance)
        )
        chunk = chunk[chunk["label"] >= 0]
        stars = (chunk["ReviewStatus"].str.strip().str.lower()
                 .map(REVIEW_STARS).fillna(0))
        chunk = chunk[stars >= MIN_REVIEW_STARS]
        if len(chunk):
            kept.append(chunk)
        log.info("  chunk %d -> running total %d", i,
                 sum(len(c) for c in kept))
    df = pd.concat(kept, ignore_index=True)
    log.info("Kept %d labeled SNVs before cleaning", len(df))
    return df


def clean_and_sample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Chromosome": "chrom", "PositionVCF": "pos",
        "ReferenceAlleleVCF": "ref", "AlternateAlleleVCF": "alt",
        "GeneSymbol": "gene",
    })
    df = df[df["ref"].isin(list("ACGT")) & df["alt"].isin(list("ACGT"))]
    df["variant_id"] = df.apply(
        lambda r: build_hgvs(r["chrom"], r["pos"], r["ref"], r["alt"]), axis=1
    )
    df = df.dropna(subset=["variant_id"]).drop_duplicates("variant_id")
    df["gene"] = df["gene"].fillna("UNKNOWN").replace("-", "UNKNOWN")
    df["is_transition"] = df.apply(
        lambda r: is_transition(r["ref"], r["alt"]), axis=1
    )

    out = df[["variant_id", "chrom", "pos", "ref", "alt", "gene",
              "is_transition", "label"]].copy()

    if len(out) > MAX_VARIANTS:
        frac = MAX_VARIANTS / len(out)
        out = (out.groupby("label", group_keys=False)
               .apply(lambda g: g.sample(frac=frac, random_state=RANDOM_SEED)))
    pos = int((out["label"] == 1).sum())
    neg = int((out["label"] == 0).sum())
    log.info("Final dataset: %d variants (%d pathogenic / %d benign)",
             len(out), pos, neg)
    return out


def main() -> None:
    download()
    df = load_and_filter()
    out = clean_and_sample(df)
    ensure_dir(DATA_DIR)
    out.to_csv(LABELED_CSV, index=False)
    log.info("Wrote -> %s", LABELED_CSV)


if __name__ == "__main__":
    main()
