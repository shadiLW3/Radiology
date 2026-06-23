"""NPI credential verification via the public NPPES registry (no auth, no key).

We BADGE, never gate — anyone can play; this just labels verified clinicians so the
leaderboard can stratify human-vs-model by real expertise.

Honest limits (surfaced in the UI): NPPES confirms an NPI is real, active, matches a
name, and carries a physician taxonomy — it does NOT prove identity (anyone can type a
real clinician's *public* NPI). We persist only the badge, matched specialty, a
name-match flag, the last 4 digits of the NPI, and a timestamp.
"""
import json
import re
import urllib.parse
import urllib.request

NPPES_URL = "https://npiregistry.cms.hhs.gov/api/"


def luhn_npi_valid(npi):
    """NPI check digit uses the Luhn algorithm over an '80840' + first-9-digits prefix."""
    if not re.fullmatch(r"\d{10}", npi or ""):
        return False
    digits = [int(c) for c in ("80840" + npi[:9])]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - (total % 10)) % 10 == int(npi[9])


def classify(codes):
    """Taxonomy codes -> our badge tier (most specific first). None = not badged."""
    if any(c.startswith("207N") for c in codes):
        return "dermatologist"
    if any(c.startswith("2085") for c in codes):
        return "radiologist"
    if any(c.startswith(("207", "208")) for c in codes):
        return "physician"
    return None


def _norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def verify(npi, first_name, last_name, timeout=15):
    npi = (npi or "").strip()
    out = {"ok": False, "badge": None, "specialty": None, "name_match": False,
           "npi_last4": npi[-4:] if len(npi) >= 4 else "", "message": ""}

    if not luhn_npi_valid(npi):
        out["message"] = "That doesn't look like a valid 10-digit NPI."
        return out
    params = urllib.parse.urlencode({"version": "2.1", "number": npi,
                                     "enumeration_type": "NPI-1"})
    try:
        with urllib.request.urlopen(f"{NPPES_URL}?{params}", timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception:
        out["message"] = "Couldn't reach the NPI registry — please try again."
        return out

    if not data.get("result_count"):
        out["message"] = "No individual provider found with that NPI."
        return out
    rec = data["results"][0]
    basic = rec.get("basic", {})
    if basic.get("status") != "A":
        out["message"] = "That NPI record is not active."
        return out

    taxonomies = rec.get("taxonomies", [])
    badge = classify([t.get("code", "") for t in taxonomies])
    if not badge:
        out["message"] = "That NPI is a real provider, but not a physician this demo badges."
        return out

    name_match = (_norm(first_name) == _norm(basic.get("first_name")) and
                  _norm(last_name) == _norm(basic.get("last_name")))
    specialty = next((t.get("desc") for t in taxonomies if t.get("primary")),
                     taxonomies[0].get("desc"))
    out.update(ok=True, badge=badge, specialty=specialty, name_match=name_match,
               message="Verified" if name_match
               else "NPI matches a real provider, but the name you entered didn't match.")
    return out
