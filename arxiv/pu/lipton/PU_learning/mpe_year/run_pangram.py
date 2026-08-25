"""
Pangram MPE for mpe_year (chunk-level; no training) -- via the BULK API.

Runs Pangram on the shared 75-word test chunks of each year (2020/2023/2025) --
the exact same chunks PU and James score -- using Pangram's bulk endpoint
(20% cheaper than realtime). For every chunk it saves EVERYTHING the API returns:
the full raw JSON per item (results/pangram_raw/pangram_bulk_{year}.json), plus a
flat per-chunk CSV with the raw JSON string and every top-level scalar field.

Bulk flow (discovered against text.external-api.pangram.com):
  submit : POST /bulk  {"items":[{"id","text"}, ...]}          -> {"bulk_id", ...}
  status : GET  /bulk/{bulk_id}                                -> {"status","succeeded","failed",...}
  results: GET  /bulk/{bulk_id}/results?offset=&limit=100      -> {"items":[{id,stage,error,result}, ...]}
each item's `result` is the same schema as PangramText.predict() (V3):
  fraction_ai, fraction_ai_assisted, fraction_human, prediction, prediction_short,
  headline, num_ai_segments, num_ai_assisted_segments, num_human_segments, windows[...]

Env: llm_master. Run explicitly (costs real API calls; ~$0.05/100 words, -20% bulk):
    cd .../PU_learning
    /home/kkr36/.conda/envs/llm_master/bin/python mpe_year/run_pangram.py
    # test first:  ... mpe_year/run_pangram.py --limit 5
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import requests

PU_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PU_ROOT not in sys.path:
    sys.path.insert(0, PU_ROOT)

from mpe_year.mpe_data import YEARS, eval_human_chunks

with open("/home/kkr36/creds.json", "r") as fh:
    PANGRAM_API_KEY = json.load(fh)["pangram_api_key"]

BULK_BASE = "https://text.external-api.pangram.com/bulk"
HEADERS = {"Content-Type": "application/json", "x-api-key": PANGRAM_API_KEY}

OUT_DIR = os.path.join(PU_ROOT, "mpe_year", "results")
RAW_DIR = os.path.join(OUT_DIR, "pangram_raw")
os.makedirs(RAW_DIR, exist_ok=True)

SUBMIT_BATCH = 500     # items per bulk submit
POLL_SECONDS = 5
POLL_TIMEOUT = 1800    # per bulk job
PAGE_LIMIT = 100
TERMINAL = {"succeeded", "failed", "partial", "completed", "error", "cancelled"}


# ── bulk API helpers ──────────────────────────────────────────────────────────
def submit_bulk(items):
    r = requests.post(BULK_BASE, json={"items": items}, headers=HEADERS, timeout=90)
    r.raise_for_status()
    j = r.json()
    return j["bulk_id"], j.get("failed_items", [])


def wait_bulk(bulk_id):
    t0 = time.time()
    while True:
        r = requests.get(f"{BULK_BASE}/{bulk_id}", headers=HEADERS, timeout=60)
        r.raise_for_status()
        j = r.json()
        done = j.get("status") in TERMINAL or \
            (j.get("succeeded", 0) + j.get("failed", 0)) >= j.get("total_items", 1e18)
        if done:
            return j
        if time.time() - t0 > POLL_TIMEOUT:
            raise TimeoutError(f"bulk {bulk_id} not done after {POLL_TIMEOUT}s: {j}")
        time.sleep(POLL_SECONDS)


def fetch_results(bulk_id):
    out, offset = [], 0
    while True:
        r = requests.get(f"{BULK_BASE}/{bulk_id}/results",
                         params={"offset": offset, "limit": PAGE_LIMIT}, headers=HEADERS, timeout=90)
        r.raise_for_status()
        j = r.json()
        items = j.get("items", [])
        out.extend(items)
        total = j.get("total_items", len(out))
        offset += PAGE_LIMIT
        if offset >= total or not items:
            return out


# ── parse: keep everything the API returns ────────────────────────────────────
def parse_result(result):
    if not isinstance(result, dict) or len(result) == 0:
        return {"pangram_ok": False, "raw_json": json.dumps(result)}
    out = {"pangram_ok": True, "raw_json": json.dumps(result)}
    for k, v in result.items():
        if k == "text":
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[f"{k}__len"] = len(v)
    windows = result.get("windows", []) or []
    if windows:
        out["window_labels"] = json.dumps([w.get("label") for w in windows])
        ai_scores = [w.get("ai_assistance_score", np.nan) for w in windows]
        out["window_ai_assistance_scores"] = json.dumps(ai_scores)
        out["window_confidences"] = json.dumps([w.get("confidence") for w in windows])
        out["window_word_counts"] = json.dumps([w.get("word_count") for w in windows])
        valid = [s for s in ai_scores if isinstance(s, (int, float)) and not np.isnan(s)]
        out["mean_window_ai_assistance"] = float(np.mean(valid)) if valid else np.nan
    return out


# ── per-year run ──────────────────────────────────────────────────────────────
def run_year(year, limit=None):
    print(f"\n===== Pangram (bulk) | {year} =====")
    chunks = eval_human_chunks(year)
    if limit:
        chunks = chunks[:limit]
    id_to_text = {f"{year}_{i}": c for i, c in enumerate(chunks)}
    items = [{"id": cid, "text": txt} for cid, txt in id_to_text.items()]
    print(f"submitting {len(items)} chunks in batches of {SUBMIT_BATCH}")

    raw_items, all_failed = [], []
    for b in range(0, len(items), SUBMIT_BATCH):
        batch = items[b:b + SUBMIT_BATCH]
        bulk_id, failed = submit_bulk(batch)
        all_failed.extend(failed)
        status = wait_bulk(bulk_id)
        got = fetch_results(bulk_id)
        raw_items.extend(got)
        print(f"  batch {b//SUBMIT_BATCH}: bulk_id={bulk_id} status={status.get('status')} "
              f"got={len(got)} failed={len(failed)}")

    # save the raw bulk items verbatim (maximum fidelity)
    raw_json_path = os.path.join(RAW_DIR, f"pangram_bulk_{year}.json")
    with open(raw_json_path, "w") as fh:
        json.dump({"year": year, "items": raw_items, "submit_failed": all_failed}, fh)

    # flat per-chunk table
    rows = []
    for it in raw_items:
        row = {"id": it.get("id"), "index": it.get("index"),
               "stage": it.get("stage"), "error": json.dumps(it.get("error"))}
        row["text"] = id_to_text.get(it.get("id"))
        row.update(parse_result(it.get("result")))
        rows.append(row)
    for f in all_failed:  # chunks Pangram rejected at submit (e.g. too short)
        rows.append({"id": f.get("id"), "stage": "SUBMIT_FAILED",
                     "error": json.dumps(f), "text": id_to_text.get(f.get("id")),
                     "pangram_ok": False})
    raw_df = pd.DataFrame(rows)
    raw_csv = os.path.join(RAW_DIR, f"pangram_raw_{year}.csv")
    raw_df.to_csv(raw_csv, index=False)
    print(f"saved -> {raw_csv}  ({len(raw_df)} rows)  and  {raw_json_path}")
    return summarize(raw_df, year)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def summarize(raw_df, year):
    ok = raw_df[raw_df.get("pangram_ok") == True] if "pangram_ok" in raw_df else raw_df
    row = {"method": "pangram", "year": year,
           "n_eval_chunks": len(raw_df), "n_scored": len(ok)}
    if "fraction_ai" in ok:
        row["mean_fraction_ai"] = float(_num(ok["fraction_ai"]).mean())
    if "fraction_ai_assisted" in ok:
        row["mean_fraction_ai_assisted"] = float(_num(ok["fraction_ai_assisted"]).mean())
    if "fraction_human" in ok:
        row["mean_fraction_human"] = float(_num(ok["fraction_human"]).mean())
    if "prediction_short" in ok:
        ps = ok["prediction_short"].astype(str).str.lower()
        row["frac_pred_ai"] = float((ps == "ai").mean())                       # strict AI label rate
        row["frac_pred_ai_or_assisted"] = float(ps.str.contains("ai").mean())  # AI or AI-assisted
    if "mean_window_ai_assistance" in ok:
        row["mean_window_ai_assistance"] = float(_num(ok["mean_window_ai_assistance"]).mean())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score only the first N chunks/year (testing)")
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    args = ap.parse_args()

    summary = pd.DataFrame([run_year(y, args.limit) for y in args.years])
    out = os.path.join(OUT_DIR, "pangram_mpe.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
