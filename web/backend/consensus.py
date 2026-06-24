"""STAPLE consensus (Warfield et al., IEEE TMI 2004) for fusing annotations.

The platform stores a human-drawn mask per attempt. STAPLE fuses the reference mask
+ all human masks into a probabilistic "consensus truth", weighting each annotator by
its estimated reliability (sensitivity/specificity) via expectation-maximization — a
fairer scoring target than a single noisy mask, and a way to surface where annotators
genuinely disagree (the inter-annotator "noise floor").
"""
import itertools

import numpy as np


def staple(masks, max_iter=40, tol=1e-5, prior=None):
    """Fuse binary masks -> (consensus_prob[H,W], sensitivities[R], specificities[R]).

    EM over a hidden true label T per pixel and each rater's sensitivity p_r =
    P(D=1|T=1) and specificity q_r = P(D=0|T=0). Computed in log-space for stability.
    """
    shape = np.asarray(masks[0]).shape
    D = np.stack([(np.asarray(m) > 0).astype(np.float64).ravel() for m in masks], 0)  # [R, P]
    R, _ = D.shape
    gamma = float(np.clip(prior if prior is not None else D.mean(), 1e-3, 1 - 1e-3))  # P(T=1)
    p = np.full(R, 0.99)   # sensitivity
    q = np.full(R, 0.99)   # specificity
    W = D.mean(0)          # init per-pixel P(T=1)
    for _ in range(max_iter):
        # E-step: weight of T=1 per pixel
        lp = D * np.log(p)[:, None] + (1 - D) * np.log1p(-p)[:, None]      # log P(D|T=1)
        lq = D * np.log1p(-q)[:, None] + (1 - D) * np.log(q)[:, None]      # log P(D|T=0)
        la = np.log(gamma) + lp.sum(0)
        lb = np.log1p(-gamma) + lq.sum(0)
        mx = np.maximum(la, lb)
        ea, eb = np.exp(la - mx), np.exp(lb - mx)
        Wn = ea / (ea + eb + 1e-12)
        # M-step: re-estimate each rater's reliability
        sw = Wn.sum()
        p = np.clip((Wn * D).sum(1) / (sw + 1e-9), 1e-4, 1 - 1e-4)
        q = np.clip(((1 - Wn) * (1 - D)).sum(1) / ((1 - Wn).sum() + 1e-9), 1e-4, 1 - 1e-4)
        delta = float(np.abs(Wn - W).max())
        W = Wn
        if delta < tol:
            break
    return W.reshape(shape), p, q


def consensus_mask(masks, threshold=0.5):
    prob, p, q = staple(masks)
    return (prob >= threshold).astype(np.uint8), prob, p, q


def mean_pairwise_dice(masks):
    """Mean pairwise Dice among masks — the inter-annotator agreement / 'noise floor'."""
    b = [(np.asarray(m) > 0) for m in masks]
    scores = []
    for a, c in itertools.combinations(b, 2):
        sa, sc = a.sum(), c.sum()
        if sa == 0 and sc == 0:
            scores.append(1.0)
        else:
            scores.append(2.0 * np.logical_and(a, c).sum() / (sa + sc))
    return float(np.mean(scores)) if scores else 1.0
