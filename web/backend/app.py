"""MedVS-AI Phase 1 — FastAPI backend for the human-vs-model loop.

Endpoints (see web/README.md):
  GET  /                      -> the play page
  GET  /api/next_case         -> an unseen case for this session
  GET  /cases/{id}/{which}.png-> image | gt | model PNGs (gt/model only meaningful post-reveal)
  POST /api/attempt           -> score the locked answer, persist, return the reveal
  GET  /api/leaderboard       -> stats grouped by self-reported badge

Design notes:
  - Anti-anchoring is enforced here: the model/GT answer is ONLY ever returned by /api/attempt
    (after the human submits), never by /api/next_case.
  - Model masks are precomputed at seed time; live play does no inference.
  - The model is segmentation-only and ABSTAINS on diagnosis (model_diagnosis = null) for now.
"""
import datetime
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import db
import npi
import scoring

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE, "..", "static")
EXPERT_DICE_BAND = [0.75, 0.81]  # expert-vs-expert range (see docs/ANNOTATION_RESEARCH.md)

app = FastAPI(title="MedVS-AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    db.init_db()


def _load_mask(path):
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/next_case")
def next_case(session_id: str):
    conn = db.get_conn()
    row = conn.execute(
        "SELECT case_id, width, height FROM cases "
        "WHERE case_id NOT IN (SELECT case_id FROM seen WHERE session_id = ?) "
        "ORDER BY RANDOM() LIMIT 1",
        (session_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return {"case_id": None}
    return {
        "case_id": row["case_id"],
        "image_url": f"/cases/{row['case_id']}/image.png",
        "width": row["width"],
        "height": row["height"],
        "expert_dice_band": EXPERT_DICE_BAND,
    }


@app.get("/cases/{case_id}/{which}.png")
def case_asset(case_id: str, which: str):
    if which not in ("image", "gt", "model"):
        raise HTTPException(404)
    path = os.path.join(db.CASES_DIR, case_id, f"{which}.png")
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


class VerifyReq(BaseModel):
    session_id: str
    npi: str
    first_name: str = ""
    last_name: str = ""


@app.post("/api/verify_npi")
def verify_npi(req: VerifyReq):
    """Badge (never gate) a clinician against the public NPPES registry."""
    res = npi.verify(req.npi, req.first_name, req.last_name)
    if res["ok"]:
        conn = db.get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO credentials VALUES (?,?,?,?,?,?)",
            (req.session_id, res["badge"], res["npi_last4"], res["specialty"],
             int(res["name_match"]), datetime.datetime.utcnow().isoformat()),
        )
        conn.commit(); conn.close()
    return res


@app.get("/api/credential")
def credential(session_id: str):
    conn = db.get_conn()
    row = conn.execute(
        "SELECT badge, specialty, npi_last4, name_match FROM credentials WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return {"badge": None}
    return {"badge": row["badge"], "specialty": row["specialty"],
            "npi_last4": row["npi_last4"], "name_match": bool(row["name_match"])}


class Attempt(BaseModel):
    session_id: str
    case_id: str
    badge: str = "layperson"
    diagnosis: str
    confidence: int = 50
    mask_png: str
    draw_ms: int = 0


@app.post("/api/attempt")
def attempt(a: Attempt):
    conn = db.get_conn()
    case = conn.execute("SELECT * FROM cases WHERE case_id = ?", (a.case_id,)).fetchone()
    if case is None:
        conn.close()
        raise HTTPException(404, "unknown case")

    gt = _load_mask(case["gt_mask_path"])
    model_mask = _load_mask(case["model_mask_path"])
    user = scoring.decode_mask_png(a.mask_png)

    you = scoring.all_metrics(user, gt)
    model = scoring.all_metrics(model_mask, gt)
    diagnosis_correct = (a.diagnosis == case["gt_diagnosis"])
    beat = you["dice"] > model["dice"]
    model_diag = case["model_diagnosis"]  # None if classifier not installed -> abstains

    # a verified NPI badge overrides the self-reported dropdown
    cred = conn.execute(
        "SELECT badge FROM credentials WHERE session_id = ?", (a.session_id,)).fetchone()
    badge = cred["badge"] if cred else a.badge
    verified = 1 if cred else 0

    conn.execute(
        "INSERT INTO attempts (session_id, case_id, badge, created_at, diagnosis, confidence, "
        "draw_ms, dice, iou, threshold_jaccard, hausdorff95, diagnosis_correct, "
        "beat_model_on_dice, model_dice, verified) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (a.session_id, a.case_id, badge, datetime.datetime.utcnow().isoformat(),
         a.diagnosis, a.confidence, a.draw_ms, you["dice"], you["iou"],
         you["threshold_jaccard"], you["hausdorff95"], int(diagnosis_correct),
         int(beat), model["dice"], verified),
    )
    conn.execute("INSERT OR IGNORE INTO seen VALUES (?,?)", (a.session_id, a.case_id))
    conn.commit()
    conn.close()

    return {
        "you": {**you, "diagnosis": a.diagnosis, "diagnosis_correct": diagnosis_correct},
        "model": {**model, "diagnosis": model_diag,
                  "diagnosis_correct": (model_diag == case["gt_diagnosis"]) if model_diag else None},
        "ground_truth": {"diagnosis": case["gt_diagnosis"]},
        "masks": {"gt_url": f"/cases/{a.case_id}/gt.png",
                  "model_url": f"/cases/{a.case_id}/model.png"},
        "beat_model_on_dice": beat,
        "agreed_with_model": (a.diagnosis == model_diag) if model_diag else None,
        "expert_dice_band": EXPERT_DICE_BAND,
    }


@app.get("/api/leaderboard")
def leaderboard():
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT badge, verified, COUNT(*) n, AVG(dice) avg_dice, "
        "AVG(diagnosis_correct) diag_acc, AVG(beat_model_on_dice) beat_rate "
        "FROM attempts GROUP BY badge, verified ORDER BY verified DESC, avg_dice DESC"
    ).fetchall()
    model_avg = conn.execute("SELECT AVG(model_dice) m FROM attempts").fetchone()["m"]
    conn.close()
    return {
        "by_badge": [
            {"badge": r["badge"], "verified": bool(r["verified"]), "n": r["n"],
             "avg_dice": round(r["avg_dice"] or 0, 4),
             "diagnosis_accuracy": round(r["diag_acc"] or 0, 4),
             "beat_model_rate": round(r["beat_rate"] or 0, 4)}
            for r in rows
        ],
        "model_avg_dice": round(model_avg, 4) if model_avg is not None else None,
        "expert_dice_band": EXPERT_DICE_BAND,
    }


# static assets (css/js) under /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
