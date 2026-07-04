"""Contribution ledger (M13) — who computed how much, from the signed receipts.

The head journals every verified per-hop receipt (M11) as a jsonl line
(DRIFT_JOURNAL). This folds that journal into a per-node tally — the raw input a
settlement / "For Tokens" payout layer would consume. It only *reads* signed
facts the network already produced; it mints nothing.

    drift ledger <journal.jsonl>            # per-node contribution table
    drift ledger <journal.jsonl> --verify   # re-check every receipt signature
    drift ledger <journal.jsonl> --csv out.csv

Contribution is measured in **layer-tokens** — layers served × tokens carried —
because holding more layers, or serving more tokens, is more work. A forged or
tampered line fails --verify and is excluded from a verified tally.
"""

from __future__ import annotations

import argparse
import csv
import sys

from .receipts import from_json, read_journal, verify_receipt


def aggregate(rows: list[dict], verified_only: bool = False) -> dict:
    """Per-node totals. `rows` are journal (json) receipts; with verified_only,
    signatures are re-checked and bad lines dropped."""
    agg: dict[str, dict] = {}
    for d in rows:
        if verified_only and not verify_receipt(from_json(d)):
            continue
        node = d["node"]
        a = agg.setdefault(node, {"tokens": 0, "layer_tokens": 0,
                                  "sessions": set(), "ranges": set()})
        span = int(d["end"]) - int(d["start"])
        a["tokens"] += 1
        a["layer_tokens"] += span
        a["sessions"].add(d["session"])
        a["ranges"].add((int(d["start"]), int(d["end"])))
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="drift ledger",
                                 description="aggregate per-node contribution from a receipt journal")
    ap.add_argument("journal", help="path to the jsonl receipt journal (DRIFT_JOURNAL)")
    ap.add_argument("--verify", action="store_true", help="re-check every receipt signature")
    ap.add_argument("--csv", metavar="PATH", help="also write the tally as CSV")
    args = ap.parse_args(argv)

    try:
        rows = read_journal(args.journal)
    except FileNotFoundError:
        print(f"no journal at {args.journal} — run a generation with DRIFT_JOURNAL set first",
              flush=True)
        return 1

    valid = invalid = 0
    if args.verify:
        for d in rows:
            if verify_receipt(from_json(d)):
                valid += 1
            else:
                invalid += 1

    agg = aggregate(rows, verified_only=args.verify)
    total_lt = sum(a["layer_tokens"] for a in agg.values())

    print(f"[ledger] {len(rows)} receipt(s) · {len(agg)} node(s)"
          + (f" · verified {valid} ok / {invalid} bad" if args.verify else ""), flush=True)
    print(f"{'node':<20} {'tokens':>8} {'layer-tokens':>13} {'share':>7} {'sessions':>9}  ranges",
          flush=True)
    table = []
    for node, a in sorted(agg.items(), key=lambda kv: -kv[1]["layer_tokens"]):
        share = (a["layer_tokens"] / total_lt) if total_lt else 0.0
        ranges = ",".join(f"[{s}:{e})" for s, e in sorted(a["ranges"]))
        print(f"{node[:18]+'…':<20} {a['tokens']:>8} {a['layer_tokens']:>13} "
              f"{share:>6.1%} {len(a['sessions']):>9}  {ranges}", flush=True)
        table.append((node, a["tokens"], a["layer_tokens"], round(share, 6),
                      len(a["sessions"]), ranges))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node_pubkey", "tokens", "layer_tokens", "share", "sessions", "ranges"])
            w.writerows(table)
        print(f"[ledger] wrote {args.csv}", flush=True)

    if args.verify and invalid:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
