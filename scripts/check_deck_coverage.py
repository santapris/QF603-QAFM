"""Check that restructuring report/oil_deck.md lost no content.

Usage:  .venv/Scripts/python.exe scripts/check_deck_coverage.py [BASE_COMMIT]   (default: 6e728ae, the deck before the restructure)

Every non-blank line of the deck at BASE_COMMIT (slide separators aside) must appear, verbatim after stripping, somewhere in the
current deck, in the main slides or in the appendix. A line that was reworded on purpose must be listed in
scripts/deck_coverage_rewrites.txt as `old line<TAB>reason`, and the check then requires that the old line is NOT silently missing:
it is reported as REWRITTEN, with the reason, so the reviewer sees every change. Exit code 1 lists any line that is missing without
an entry. Slides are also counted, so a dropped slide cannot hide behind unchanged lines.
"""
from __future__ import annotations

import subprocess
import sys

from oilrv.config import ROOT

sys.stdout.reconfigure(encoding="utf-8")

BASE = sys.argv[1] if len(sys.argv) > 1 else "6e728ae"
DECK = ROOT / "report" / "oil_deck.md"
ALLOW = ROOT / "scripts" / "deck_coverage_rewrites.txt"


def lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip() and ln.strip() != "---"]


def headings(text: str) -> list[str]:
    return [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.startswith("## ") or ln.startswith("# ")]


old = subprocess.run(["git", "show", f"{BASE}:report/oil_deck.md"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
new = DECK.read_text(encoding="utf-8")
new_set = set(lines(new))
rewrites = {}
if ALLOW.exists():
    for ln in ALLOW.read_text(encoding="utf-8").splitlines():
        if "\t" in ln and not ln.startswith("# old line"):
            a, b = ln.split("\t", 1)
            rewrites[a.strip()] = b.strip()

missing = [ln for ln in lines(old) if ln not in new_set]
unexplained = [ln for ln in missing if ln not in rewrites]
explained = [ln for ln in missing if ln in rewrites]
stale = [a for a in rewrites if a in new_set]
old_h, new_h = headings(old), set(headings(new))
lost_slides = [h for h in old_h if h not in new_h]
print(f"{len(lines(old))} content lines in the deck at {BASE}; {len(lines(old)) - len(missing)} present verbatim, "
      f"{len(explained)} rewritten with a reason, {len(unexplained)} unexplained.")
print(f"{len(old_h)} headings then, {len(headings(new))} now; {len(lost_slides)} headings changed or removed.")
for h in lost_slides:
    print(f"  HEADING CHANGED {h}")
for ln in explained:
    print(f"  REWRITTEN {ln[:110]}  ->  {rewrites[ln]}")
for ln in unexplained:
    print(f"  MISSING {ln}")
for a in stale:
    print(f"  NOTE: listed as rewritten but still present: {a[:90]}")
sys.exit(1 if unexplained else 0)
