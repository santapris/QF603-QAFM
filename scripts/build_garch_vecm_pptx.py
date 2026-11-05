"""GARCH and VECM slides, in the master deck's style, to paste into "Oil Strategy Selection (Draft).pptx".

Usage:  PYTHONPATH=<dir with python-pptx> .venv/Scripts/python.exe scripts/build_garch_vecm_pptx.py
        (python-pptx is not in requirements.txt; install it anywhere, e.g. pip install --target <dir> python-pptx)
Writes report/garch_vecm_slides.pptx: 4 presentation slides (Steps 8, 8b, 9, 9b) and 4 appendix slides.

The file is built FROM the master deck (its theme, slide master and "Content" layout), with the master's own slides
removed, so pasted slides keep the same fonts, colours and placeholders. Figures: report/deck_figures/19-24 (from
scripts/build_garch_vecm_figures.py). Every number is the one in report/oil_deck.md, which scripts/check_deck.py ties to the
output tables. Speaker notes carry the full deck text for each slide.
"""
from __future__ import annotations

import re
import sys

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from oilrv.config import ROOT

sys.stdout.reconfigure(encoding="utf-8")
MASTER = ROOT / "report" / "Oil Strategy Selection (Draft).pptx"
OUT = ROOT / "report" / "garch_vecm_slides.pptx"
FIG = ROOT / "report" / "deck_figures"
NAVY, BORDER, HEAD, GRID = RGBColor(0x14, 0x21, 0x3D), RGBColor(0xE3, 0xE1, 0xD8), RGBColor(0xE5, 0xE4, 0xDF), "C5C4C0"
X0, W_FULL = 0.60, 12.11


# --------------------------------------------------------------------------- helpers
def fresh_from_master() -> Presentation:
    p = Presentation(str(MASTER))
    lst = p.slides._sldIdLst
    for sid in list(lst):
        p.part.drop_rel(sid.get(qn("r:id")))
        lst.remove(sid)
    # the master's notes master has no body placeholder, so notes cannot be written; let python-pptx make a standard one
    nm = p.part._element.find(qn("p:notesMasterIdLst"))
    if nm is not None:
        for e in nm:
            p.part.drop_rel(e.get(qn("r:id")))
        p.part._element.remove(nm)
    return p


def runs(par, text: str, size: float, color=None, bold_all=False):
    """Add text to a paragraph; **x** is bold."""
    for i, chunk in enumerate(re.split(r"\*\*", text)):
        if not chunk:
            continue
        r = par.add_run()
        r.text = chunk
        r.font.size = Pt(size)
        r.font.bold = bold_all or i % 2 == 1
        if color is None:                                  # the master's body colour, theme tx2
            sf = etree.SubElement(r.font._rPr, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:schemeClr"), val="tx2")
        else:
            r.font.color.rgb = color


def textbox(slide, x, y, w, h, paras, size=13.0, color=None, space_after=6, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, t in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.space_after = Pt(space_after)
        runs(par, t, size, color)
    return tb


def label(slide, x, y, w, text):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.3))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run()
    r.text = text
    r.font.size, r.font.bold = Pt(12), True
    r.font._rPr.set("spc", "100")
    sf = etree.SubElement(r.font._rPr, qn("a:solidFill"))
    etree.SubElement(sf, qn("a:schemeClr"), val="accent1")


def card(slide, x, y, w, h, name="card"):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    s.adjustments[0] = 0.02
    s.fill.solid()
    s.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    s.line.color.rgb = BORDER
    s.line.width = Emu(6350)
    s.shadow.inherit = False
    return s


def text_card(slide, x, y, w, h, title, paras, size=13.0):
    card(slide, x, y, w, h, f"{title.title()} card")
    label(slide, x + 0.2, y + 0.12, w - 0.4, title)
    textbox(slide, x + 0.2, y + 0.48, w - 0.4, h - 0.58, paras, size)


def picture_card(slide, path, x, y, w, h, alt):
    """Card of the given box; the picture is fitted inside it, centred, aspect kept."""
    card(slide, x, y, w, h, "chart card")
    iw, ih = Image.open(path).size
    bw, bh = w - 0.16, h - 0.16
    pw, ph = (bw, bw * ih / iw) if bw * ih / iw <= bh else (bh * iw / ih, bh)
    pic = slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def table(slide, x, y, w, rows, col_w, size=12.0, row_h=0.34, bold_first_col=True):
    n, k = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n, k, Inches(x), Inches(y), Inches(w), Inches(row_h * n))
    tbl = gf.table
    tp = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tp.set(a, "0")
    sid = tp.find(qn("a:tableStyleId"))
    if sid is not None:
        tp.remove(sid)
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(w * cw / tot)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Emu(73152)
            c.margin_top = c.margin_bottom = Emu(27432)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            par = tf.paragraphs[0]
            par.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            bold = i == 0 or (bold_first_col and j == 0)
            for t_i, chunk in enumerate(re.split(r"\*\*", str(val))):
                if chunk:
                    r = par.add_run()
                    r.text = chunk
                    r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold or t_i % 2 == 1, NAVY
            tcPr = c._tc.get_or_add_tcPr()
            for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(side), w="6350", cap="flat", cmpd="sng", algn="ctr")
                sf = etree.SubElement(ln, qn("a:solidFill"))
                etree.SubElement(sf, qn("a:srgbClr"), val=GRID)
            sf = etree.SubElement(tcPr, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr"), val="E5E4DF" if i == 0 else "FFFFFF")
    return gf


def new_slide(p, kicker, title, notes):
    s = p.slides.add_slide(p.slide_layouts.get_by_name("Content"))
    for ph in s.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 100:
            ph.text_frame.text = kicker
        elif idx == 101:
            ph.text_frame.text = title
    s.notes_slide.notes_text_frame.text = notes
    return s


# --------------------------------------------------------------------------- deck text, for the notes
DECK = (ROOT / "report" / "oil_deck.md").read_text(encoding="utf-8")


def section(heading: str) -> str:
    """The deck's text for one slide, markdown stripped to plain text, for the speaker notes."""
    i = DECK.index(heading)
    j = DECK.find("\n---", i)
    body = DECK[i:j]
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
    body = re.sub(r"</?su[bp]>", "", body).replace("**", "")
    return "\n".join(ln for ln in body.splitlines() if ln.strip())


# --------------------------------------------------------------------------- slides
def build():
    p = fresh_from_master()

    # ---- Step 8 -------------------------------------------------------------------------------------------
    s = new_slide(p, "STEP 8 OF 9 · GARCH", "We can't forecast the direction, but we can forecast the size",
                  section("## Step 8: we cannot forecast"))
    picture_card(s, FIG / "19_garch_band.png", X0, 1.45, W_FULL, 4.05,
                 "HO minus RB daily moves 2021 to 2023 with the GARCH plus-minus two sigma band and the frozen 3-year band; "
                 "bars of how often a day breaks each band on stress and calm days.")
    text_card(s, X0, 5.62, 6.95, 1.28, "WHAT THE CHART SHOWS", [
        "From 24 Feb 2022 the frozen 3-year band is still set by calm years, so HO − RB breaks it almost daily (red dots). "
        "The GARCH band widens the day after the first big move.",
        "Stress days outside ±2σ: **frozen sd 14–15%, GARCH 6–7%** (a normal curve: 4.6%)."], size=12)
    text_card(s, X0 + 7.1, 5.62, W_FULL - 7.1, 1.28, "DECISION", [
        "σ̂ is tomorrow's expected move size. A shock halves in **26–38 trading days**, and GARCH beats the 3-year sd out of "
        "sample on all three spreads (t −2.6 to −4.9). Every position is sized on σ̂."], size=12)

    # ---- Step 8b ------------------------------------------------------------------------------------------
    s = new_slide(p, "STEP 8 OF 9 · GARCH SIZING", "Size on the forecast, so every day risks about the same dollars",
                  section("## Step 8b: size on the forecast"))
    picture_card(s, FIG / "20_garch_sizing_chain.png", X0, 1.45, 7.75, 5.45,
                 "HO minus RB 2019 to 2026: forecast volatility, implied units and risk taken under GARCH and the frozen 3-year sd.")
    rx, rw = X0 + 7.95, W_FULL - 7.95
    text_card(s, rx, 1.45, rw, 0.95, "THE RULE", ["units = **$6,000 / (σ̂ × $ per unit)**"], size=13)
    label(s, rx, 2.52, rw, "SAME TRADES, ONLY THE SIZING CHANGES")
    table(s, rx, 2.86, rw, [["Net, 2010–26", "Sharpe", "Max drawdown"],
                            ["3-2-1 crack", "0.36 → 0.32", "−$159k → −$156k"],
                            ["Brent − WTI", "−0.55 → −0.47", "−$321k → −$307k"],
                            ["HO − RB", "−0.07 → +0.02", "**−$356k → −$182k**"],
                            ["Crack, Johansen", "−0.27 → −0.17", "−$224k → −$159k"]], [1.25, 1.05, 1.5], size=10.5, row_h=0.3)
    text_card(s, rx, 4.5, rw, 2.4, "DECISION", [
        "After 24 Feb 2022 GARCH halves the units in **6 trading days**; the frozen sd takes **89**.",
        "Risk above twice the budget: **0.4–0.9%** of days with GARCH, **5.7–6.3%** with the frozen sd.",
        "Risk control, not a signal: it halves HO − RB's drawdown and cannot turn a losing signal into a winner."], size=11.5)

    # ---- Step 9 -------------------------------------------------------------------------------------------
    s = new_slide(p, "STEP 9 OF 9 · VECM", "When a gap opens, which leg closes it? In the crack, crude does",
                  section("## Step 9: when the gap opens"))
    picture_card(s, FIG / "21_vecm_who_moves.png", X0, 1.45, W_FULL, 4.05,
                 "Schematic of a VECM; t-statistics of each leg's error-correction term; share of the gap each side closed within 63 days.")
    text_card(s, X0, 5.62, 5.0, 1.28, "HOW TO READ THIS", [
        "α = the share of yesterday's gap a leg closes today. (b) is the model, daily; (c) asks the same question of the data, 63 days on."], size=12)
    text_card(s, X0 + 5.15, 5.62, W_FULL - 5.15, 1.28, "WHAT WE FOUND", [
        "**Crack:** crude closes ~53% of a gap in 63 days while the products move ~30% further away. **HO − RB:** ULSD (~27%). "
        "**Brent − WTI:** no clear leader. Decision: trade the spread as one unit, never one leg."], size=12)

    # ---- Step 9b ------------------------------------------------------------------------------------------
    s = new_slide(p, "STEP 9 OF 9 · VECM IMPLICATIONS", "At the average exit, 74–90% of the gap is still open",
                  section("## Step 9b: what the VECM means"))
    picture_card(s, FIG / "22_vecm_gap_decay.png", X0, 1.45, W_FULL, 3.95,
                 "Share of a gap still open by days held: the VECM curve, the measured local projection with 95 percent bands, and the GARCH volatility decay.")
    table(s, X0, 5.52, 7.45, [["VECM / measured", "Half-life", "At exit (21–31 d)", "3 months", "1 year"],
                              ["3-2-1 crack", "70 days", "74% / 90%", "54% / 77%", "8% / 19%"],
                              ["Brent / WTI", "77 days", "81% / 90%", "57% / 74%", "10% / **57%**"],
                              ["HO / RB", "96 days", "83% / 80%", "64% / 62%", "16% / 8%"]], [1.45, 0.95, 1.4, 1.1, 1.0], size=11, row_h=0.345)
    text_card(s, X0 + 7.6, 5.52, W_FULL - 7.6, 1.38, "DECISION", [
        "Daily mean reversion exits too early. Volatility fades (26–38 days) faster than the gap (70–96), so size by σ̂, "
        "not by how wide the gap is."], size=11.5)

    # ---- Appendix: GARCH in full --------------------------------------------------------------------------
    s = new_slide(p, "APPENDIX · GARCH IN FULL", "Parameters by era, and whether the forecast did its job",
                  section("## Test 6b: GARCH in full"))
    label(s, X0, 1.45, 6, "MEDIAN FIT BY ERA (ROLLING 756 DAYS, REFIT MONTHLY)")
    table(s, X0, 1.8, W_FULL, [["a / b / ν", "3-2-1 crack", "Brent − WTI", "HO − RB", "Vol-shock half-life (crack / B−W / HO−RB)"],
                               ["2007–13", "0.14 / 0.85 / 113", "0.15 / 0.84 / 10", "0.11 / 0.87 / 26", "99 / 48 / 40 days"],
                               ["2014–19", "0.09 / 0.83 / 18", "0.08 / 0.88 / 21", "0.08 / 0.89 / 14", "9 / 16 / 20 days"],
                               ["2020–26", "0.11 / 0.87 / 10", "0.15 / 0.83 / 10", "0.10 / 0.88 / 12", "26 / 36 / 38 days"]],
          [1.1, 1.7, 1.7, 1.7, 3.0], size=12, row_h=0.36)
    label(s, X0, 3.42, 8, "EACH DAY'S MOVE DIVIDED BY THE FORECAST")
    table(s, X0, 3.77, W_FULL, [["Spread", "Kurtosis: window sd → GARCH", "Lag-1 ACF of squared move (converged fits)", "QLIKE gain (t)"],
                                ["3-2-1 crack", "13.4 → 5.4", "0.31 → 0.04", "−0.43 (−4.9)"],
                                ["Brent − WTI", "101.6 → 6.2", "0.19 → 0.04", "−1.15 (−2.6)"],
                                ["HO − RB", "79.2 → 5.9", "0.39 → 0.11", "−0.76 (−2.7)"]],
          [1.3, 2.4, 3.2, 1.8], size=12, row_h=0.36)
    text_card(s, X0, 5.42, W_FULL, 1.48, "HOW TO READ THIS", [
        "a + b near 1: storms persist. ν of 10–26: fat tails. Half-life = ln 2 / (−ln(a + b)). The fit converges in 86%, 78% and 84% "
        "of months; otherwise σ̂ falls back to the window sd. Clustering is reduced, not removed: Ljung–Box on squared moves still rejects on the spreads."],
        size=12)

    # ---- Appendix: GARCH forecast -------------------------------------------------------------------------
    s = new_slide(p, "APPENDIX · GARCH FORECAST", "GARCH rises into the shock; the frozen sd lags by months",
                  section("## Test 6: can't predict direction"))
    picture_card(s, FIG / "23_garch_forecast_vs_realised.png", X0, 1.45, W_FULL, 3.75,
                 "GARCH forecast, frozen window sd and realised volatility for the three spreads, 2020 to 2022.")
    text_card(s, X0, 5.35, W_FULL, 1.55, "WHAT THE CHART SHOWS", [
        "In 2020 and 2022 the GARCH forecast tracks realised volatility within days; the frozen sd steps up a month at a time. "
        "Brent − WTI's flat green stretch in 2020–21 is a run of months where the fit failed and σ̂ fell back to a constant.",
        "Same gated trades, only the units swapped (gross): HO − RB's drawdown falls from −$330k to −$157k; the crack's rises slightly (−$139k to −$158k)."],
        size=12)

    # ---- Appendix: VECM in full ---------------------------------------------------------------------------
    s = new_slide(p, "APPENDIX · VECM IN FULL", "Three specifications: crude in the crack, diesel in HO − RB",
                  section("## Test 7b: VECM in full"))
    label(s, X0, 1.45, 7.6, "KNOWN VECTOR: LOGS (FULL SAMPLE) AND DOLLARS (ROLLING 3 YEARS)")
    table(s, X0, 1.8, 7.55, [["Spread: leg", "Logs: α a day (t)", "Rolling: corrects in"],
                             ["Crack: WTI", "**+0.017 (2.6)**", "**16%**"],
                             ["Crack: RBOB", "+0.007 (1.6), wrong sign", "0.5%"],
                             ["Crack: ULSD", "+0.007 (1.8), wrong sign", "2%"],
                             ["Brent / WTI: WTI", "+0.012 (1.3)", "**29%**"],
                             ["Brent / WTI: Brent", "+0.003 (0.4), wrong sign", "5%"],
                             ["HO / RB: ULSD", "−0.007 (−1.5)", "**24%**"],
                             ["HO / RB: RBOB", "−0.000 (−0.0), wrong sign", "9%"],
                             ["Gap on itself", "−0.010 / −0.009 / −0.007", "half-lives 70 / 77 / 96 d"]],
          [2.0, 2.7, 2.3], size=11.5, row_h=0.355)
    rx, rw = X0 + 7.75, W_FULL - 7.75
    label(s, rx, 1.45, rw, "UNRESTRICTED JOHANSEN, DOLLARS")
    table(s, rx, 1.8, rw, [["α (t)", "Vector 1", "Vector 2"],
                           ["WTI", "−0.0008 (−1.2)", "−0.0017 (−1.1)"],
                           ["RBOB", "**−0.0022 (−2.9)**", "−0.0046 (−2.6)"],
                           ["ULSD", "−0.0008 (−1.0)", "−0.0007 (−0.4)"],
                           ["Vector", "CL − 1.74 HO", "RB − 0.05 HO"]], [0.9, 1.6, 1.6], size=10.5, row_h=0.34)
    text_card(s, rx, 3.68, rw, 3.22, "HOW TO READ THIS", [
        "Gap = ln A − ln B: the first leg corrects with α < 0, the second with α > 0.",
        "The unrestricted VECM picks RBOB, but its vectors are not the traded crack.",
        "Local projection, 63 days: crack crude +53% (±53), products −30% (±45); ULSD +27% (±33)."], size=11.5)

    # ---- Appendix: VECM rolling ---------------------------------------------------------------------------
    s = new_slide(p, "APPENDIX · VECM ROLLING", "Rolling crack loadings: crude corrects most often",
                  section("## Test 7: when the gap is off"))
    picture_card(s, FIG / "24_vecm_rolling.png", X0, 1.45, 8.6, 5.45,
                 "Rolling t-statistics of each crack leg's VECM loading with the vector fixed at 3:2:1, and the Johansen variant's weights.")
    text_card(s, X0 + 8.75, 1.45, W_FULL - 8.75, 5.45, "WHAT THE CHART SHOWS", [
        "Top: t-statistic of each leg's loading, rolling 3 years. CL corrects when t > +2; RB and HO when t < −2.",
        "CL crosses +2 in 16% of windows; RB and HO cross −2 in 0.5% and 2%. In 2011, 2022 and 2026 all three legs jump together: a common shock, not a correction.",
        "Bottom: the free Johansen vector is usable in only 30% of months, and its weights jump: one more reason not to trade a single leg."],
        size=12)

    p.save(str(OUT))
    print("wrote", OUT.relative_to(ROOT), "with", len(p.slides), "slides")


if __name__ == "__main__":
    build()
