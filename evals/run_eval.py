"""Run the 30-question evaluation against the local gold layer and write a report.

Usage:
    uv run python evals/run_eval.py                 # all cases, Groq first then OpenRouter
    LLM_MODE=groq uv run python evals/run_eval.py   # one provider only
    uv run python evals/run_eval.py --ids rev_2010 champions

Exit code is 1 when fewer than --pass-mark cases pass (default 27/30).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tillwise.agent.graph import ask, build_graph  # noqa: E402
from tillwise.agent.grounding import extract_numbers  # noqa: E402
from tillwise.agent.store import DuckDBStore  # noqa: E402
from tillwise.agent.tools import Toolbox  # noqa: E402

REFUSAL = re.compile(
    r"can['’]?t|cannot|can not|don['’]?t have|do not have|no data|not available|isn['’]?t available|outside|"
    r"only covers|only cover|unable|not able|not possible|read[- ]only|only read|doesn['’]?t (?:include|contain|cover)|"
    r"does not (?:include|contain|cover)|not (?:one of|covered|included)|no (?:forecast|sales|record)",
    re.IGNORECASE)


def number_found(answer: str, value: float, pct: bool) -> bool:
    for v, _ in extract_numbers(answer):
        if pct and abs(v - value) <= 0.15:
            return True
        if not pct and abs(v - value) <= 0.01 * max(abs(value), 1.0):
            return True
    return False


def score_case(case: dict, out: dict, store: DuckDBStore) -> tuple[bool, list[str]]:
    answer = str(out["answer"])
    notes = []
    if case.get("refuse"):
        # Numbers echoed from the question (stock codes, customer IDs) are not invented figures.
        asked = {v for v, _ in extract_numbers(case["question"])}
        big_numbers = [v for v, pct in extract_numbers(answer)
                       if not pct and v >= 100 and not 1900 <= v <= 2100 and v not in asked]
        refused = bool(REFUSAL.search(answer))
        if not refused:
            notes.append("did not decline")
        if big_numbers:
            notes.append(f"stated numbers {big_numbers[:3]}")
        return refused and not big_numbers, notes

    rows = store.query(case["truth_sql"])
    if not rows:
        return False, ["truth query returned no rows"]
    ok = True
    for chk in case["checks"]:
        targets = rows if chk.get("all_rows") else rows[:1]
        for row in targets:
            val = row[chk["col"]]
            if chk["type"] == "number":
                if val is None or not number_found(answer, float(val), chk.get("pct", False)):
                    ok = False
                    notes.append(f"missing {chk['col']}={float(val):,.2f}" if val is not None else "truth is null")
            else:
                alts = [str(val)]
                if chk.get("alt_sql_desc"):
                    d = store.query("SELECT description FROM gold.dim_product WHERE stock_code = :c AND is_current",
                                    {"c": val})
                    alts += [r["description"] for r in d if r["description"]]
                if not any(a.lower() in answer.lower() for a in alts):
                    ok = False
                    notes.append(f"missing text {alts[0]!r}")
    return ok, notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--pass-mark", type=int, default=27)
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between cases (free-tier rate limits)")
    args = parser.parse_args()

    cases = yaml.safe_load((ROOT / "evals" / "cases.yaml").read_text(encoding="utf-8"))
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    store = DuckDBStore()
    graph = build_graph(Toolbox(store))

    results, passed = [], 0
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        try:
            out = ask(graph, case["question"])
            ok, notes = score_case(case, out, store)
        except Exception as exc:  # noqa: BLE001
            out, ok, notes = {"answer": "", "tool_calls": [], "provider": None}, False, [f"error: {exc!r}"[:300]]
        passed += ok
        results.append({"id": case["id"], "question": case["question"], "pass": ok, "notes": notes,
                        "answer": out.get("answer"), "tools": [t["tool"] for t in out.get("tool_calls", [])],
                        "provider": out.get("provider"), "ungrounded": out.get("ungrounded_numbers"),
                        "seconds": round(time.time() - t0, 1)})
        mark = "PASS" if ok else "FAIL"
        print(f"[{i:2d}/{len(cases)}] {mark} {case['id']:28s} {out.get('provider') or '-':10s} "
              f"{'; '.join(notes)}", flush=True)
        time.sleep(args.pause)

    total = len(cases)
    providers = sorted({r["provider"] for r in results if r["provider"]})
    print(f"\nScore: {passed}/{total}  (pass mark {args.pass_mark})  providers: {', '.join(providers)}")

    reports = ROOT / "evals" / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    (reports / f"eval-{stamp}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    lines = [f"# Eval {stamp}: {passed}/{total}", "", "| Case | Result | Tools | Provider | Notes |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {'✅' if r['pass'] else '❌'} | {', '.join(r['tools'])} | {r['provider'] or ''} "
                     f"| {'; '.join(r['notes'])} |")
    (reports / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed >= min(args.pass_mark, total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
