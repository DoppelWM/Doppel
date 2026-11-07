"""Aggregate the JSON lines of scripts/eval/evaluate.py into a table (mean +/- standard error per group).

    python scripts/eval/summarize.py results/*.jsonl [--by evalset] [--csv summary.csv]

Columns: PSNR / SSIM / LPIPS over predicted frames 1-16; ``scs_objects`` = the SCS reported in the
paper's whole-frame tables (averaged over the non-embodiment objects); ``scs_arm`` (the embodiment's
own mask); ``scs_dynamic`` (non-embodiment objects that move in the ground truth); ``scs_all`` (all
tracked masks, embodiment included; not a paper number).

The paper reports validation and test separately: group by ``evalset`` (the default). Pass the
results of one model per call: rows are de-duplicated on (row_key, group).
"""
import argparse
import json

import numpy as np
import pandas as pd

METRICS = ["psnr", "ssim", "lpips", "scs_objects", "scs_arm", "scs_dynamic", "scs_all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--by", default="evalset", help="grouping column (evalset, dataset, robot_type)")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for p in a.jsonl for l in open(p) if l.strip()]
    df = pd.DataFrame(rows)
    if "scs" in df:
        df = df.rename(columns={"scs": "scs_all"})
    df = df.drop_duplicates(["row_key", a.by])
    out = []
    for g, d in list(df.groupby(a.by)) + [("ALL", df)]:
        rec = {a.by: g, "n": len(d)}
        for m in METRICS:
            if m in d:
                v = d[m].dropna().to_numpy(float)
                rec[m] = float(v.mean()) if len(v) else np.nan
                rec[m + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        out.append(rec)
    t = pd.DataFrame(out)
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.precision", 4):
        print(t.to_string(index=False))
    if a.csv:
        t.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
