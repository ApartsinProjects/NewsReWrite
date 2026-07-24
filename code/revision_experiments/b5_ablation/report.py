"""B5: Reformat b2 summary.json into a compact ablation table.

Reads results/b2/summary.json, writes results/b5/ablation_table.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import RESULTS_DIR, ensure_dirs


def _cell_label(a: float, b: float) -> str:
    on = lambda x: "on" if float(x) > 0 else "off"
    return f"tac {on(a)} / cb {on(b)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-json", default=str(RESULTS_DIR / "b2" / "summary.json"))
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "b5" / "ablation_table.md"))
    args = ap.parse_args()

    ensure_dirs()
    data = json.loads(Path(args.in_json).read_text())
    out = Path(args.out_md); out.parent.mkdir(parents=True, exist_ok=True)

    rows = data.get("conditions", [])
    metrics = list(rows[0]["metrics"].keys()) if rows else []

    lines = ["# B5 ablation, guide-on / guide-off", "",
             "| condition | alpha | beta | n | " + " | ".join(metrics) + " |",
             "|---|---|---|---|" + "---|" * len(metrics)]
    for c in rows:
        cells = [_cell_label(c["alpha"], c["beta"]), f"{c['alpha']}", f"{c['beta']}", f"{c['n']}"]
        for m in metrics:
            v = c["metrics"][m]
            cells.append(f"{v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
        lines.append("| " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[b5] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
