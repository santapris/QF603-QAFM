"""GARCH and VECM slides, in the oil deck's style, to paste into report/oil_deck.pptx.

Usage:  PYTHONPATH=<dir with python-pptx> .venv/Scripts/python.exe scripts/build_garch_vecm_pptx.py
        (python-pptx is not in requirements.txt; install it anywhere, e.g. pip install --target <dir> python-pptx)
Writes report/garch_vecm_slides.pptx: 4 presentation slides (Steps 8, 8b, 9, 9b) and 4 appendix slides.

The file is built FROM report/oil_deck.pptx (its theme, slide master and "CONTENT" layout, with the footer and slide number),
with that deck's own slides removed, and copies its look: white page, navy serif title, Calibri 14 body with bullets, navy-header
tables with light banding, and the cream "For a trade" callout. Figures: report/deck_figures/19-24 (from
scripts/build_garch_vecm_figures.py). Every number is the one in report/oil_deck.md, which scripts/check_deck.py ties to the
output tables. Speaker notes carry the full deck text for each slide.
"""
from __future__ import annotations

import copy
import re
import sys

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from oilrv.config import ROOT

sys.stdout.reconfigure(encoding="utf-8")
BASE = ROOT / "report" / "oil_deck.pptx"
OUT = ROOT / "report" / "garch_vecm_slides.pptx"
FIG = ROOT / "report" / "deck_figures"
INK, AMBER, HEADER, BAND, GRID, CREAM = "1A1F24", "C77700", "13303B", "F3F6F7", "D3DBDF", "FBF0DD"
X0, W = 0.60, 12.13                                       # the oil deck's content box (inches)


# --------------------------------------------------------------------------- helpers
def fresh_from_base():
    p = Presentation(str(BASE))
    # the slide-number placeholder, copied onto every new slide (the layout's footer text comes for free)
    sldnum = next(sh for sh in p.slides[4].shapes if sh.name.startswith("Slide Number"))._element
    lst = p.slides._sldIdLst
    for sid in list(lst):
        p.part.drop_rel(sid.get(qn("r:id")))
        lst.remove(sid)
    nm = p.part._element.find(qn("p:notesMasterIdLst"))   # let python-pptx make a standard notes master
    if nm is not None:
        for e in nm:
            p.part.drop_rel(e.get(qn("r:id")))
        p.part._element.remove(nm)
    return p, sldnum


def add_runs(par, text: str, size: float, color: str = INK, bold: bool = False):
    """Add text to a paragraph; **x** is bold. Calibri, as in the oil deck."""
    for i, chunk in enumerate(re.split(r"\*\*", text)):
        if not chunk:
            continue
        r = par.add_run()
        r.text = chunk
        r.font.size = Pt(size)
        r.font.bold = bold or i % 2 == 1
        r.font.name = "Calibri"
        r.font.color.rgb = RGBColor.from_string(color)


def textbox(slide, x, y, w, h, paras, size=13.0, bullets=True, space_after=6, name="Body text"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, t in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.space_after = Pt(space_after)
        pPr = par._p.get_or_add_pPr()
        if bullets:
            pPr.set("marL", "228600")
            pPr.set("indent", "-228600")
            etree.SubElement(pPr, qn("a:buSzPct"), val="100000")
            etree.SubElement(pPr, qn("a:buChar"), char="•")
        else:
            pPr.set("marL", "0")
            pPr.set("indent", "0")
            etree.SubElement(pPr, qn("a:buNone"))
        add_runs(par, t, size)
    return tb


def callout(slide, text: str, y=6.05, h=0.70):
    """The deck's cream 'For a trade' band."""
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(X0), Inches(y), Inches(W), Inches(h))
    s.name = "For a trade card"
    s.adjustments[0] = 0.10
    s.fill.solid()
    s.fill.fore_color.rgb = RGBColor.from_string(CREAM)
    s.line.color.rgb = RGBColor.from_string(CREAM)
    s.shadow.inherit = False
    tb = slide.shapes.add_textbox(Inches(X0 + 0.25), Inches(y + 0.08), Inches(W - 0.5), Inches(h - 0.16))
    tb.name = "For a trade text"
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    par = tf.paragraphs[0]
    add_runs(par, "For a trade   ", 14, AMBER, bold=True)
    add_runs(par, text, 14)


def caption(slide, x, y, w, text):
    textbox(slide, x, y, w, 0.3, [f"**{text}**"], size=13, bullets=False, space_after=0, name="Caption")


def picture(slide, path, x, y, w, alt, max_h=None):
    """Picture of width w (inches), shrunk to max_h if it would be taller, then centred in w; returns its bottom edge."""
    iw, ih = Image.open(path).size
    pw, ph = w, w * ih / iw
    if max_h and ph > max_h:
        pw, ph = max_h * iw / ih, max_h
    pic = slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y), Inches(pw), Inches(ph))
    pic.name = "Figure"
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return y + ph


def table(slide, x, y, w, rows, col_w, size=12.0, row_h=0.32):
    n, k = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n, k, Inches(x), Inches(y), Inches(w), Inches(row_h * n))
    gf.name = "Table"
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
            c.margin_left = c.margin_right = Emu(63500)
            c.margin_top = c.margin_bottom = Emu(38100)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.text_frame.word_wrap = True
            add_runs(c.text_frame.paragraphs[0], str(val), size, "FFFFFF" if i == 0 else INK, bold=i == 0)
            tcPr = c._tc.get_or_add_tcPr()
            for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(side), w="9525", cap="flat", cmpd="sng", algn="ctr")
                sf = etree.SubElement(ln, qn("a:solidFill"))
                etree.SubElement(sf, qn("a:srgbClr"), val=GRID)
            sf = etree.SubElement(tcPr, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr"), val=HEADER if i == 0 else (BAND if i % 2 == 0 else "FFFFFF"))
    return gf


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
    p, sldnum = fresh_from_base()

    def new_slide(title: str, notes: str):
        s = p.slides.add_slide(p.slide_layouts.get_by_name("CONTENT"))
        tf = next(ph for ph in s.placeholders if ph.placeholder_format.idx == 100).text_frame
        tf.text = title
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0     # as on the oil deck's slides
        tf.vertical_anchor = MSO_ANCHOR.TOP
        s.shapes._spTree.append(copy.deepcopy(sldnum))
        s.notes_slide.notes_text_frame.text = notes
        return s

    # ---- Step 8 -------------------------------------------------------------------------------------------
    s = new_slide("Step 8: we can forecast the size of a move (GARCH)", section("## Step 8: we cannot forecast"))
    b = picture(s, FIG / "19_garch_band.png", X0, 1.35, W,
                "HO minus RB daily moves 2021 to 2023 with the GARCH plus-minus two sigma band and the frozen 3-year band; "
                "bars of how often a day breaks each band on stress and calm days.", max_h=3.4)
    textbox(s, X0, b + 0.1, W, 0.7, [
        "**In plain English.** We cannot say whether tomorrow's move is up or down, but after a stormy day another is likely. "
        "On 24 Feb 2022 the frozen 3-year band is still set by calm years, so HO − RB breaks it almost daily (red dots); the GARCH band widens the day after.",
        "On stress days a move breaks the frozen ±2 sd band **14–15%** of the time, the GARCH band **6–7%** (a normal curve: 4.6%). "
        "A shock halves in **26–38 trading days**; GARCH beats the 3-year sd out of sample on all three spreads (t −2.6 to −4.9)."],
        size=12, space_after=3)
    callout(s, "σ̂ is tomorrow's expected move size. It is the input to every position size in this deck.")

    # ---- Step 8b ------------------------------------------------------------------------------------------
    s = new_slide("Step 8b: size on the forecast, so each day risks the same", section("## Step 8b: size on the forecast"))
    picture(s, FIG / "20_garch_sizing_chain.png", X0, 1.35, 6.7,
            "HO minus RB 2019 to 2026: forecast volatility, implied units and risk taken under GARCH and the frozen 3-year sd.", max_h=4.5)
    rx, rw = X0 + 6.95, W - 6.95
    caption(s, rx, 1.35, rw, "Same trades, only the sizing changes (net, 2010–26)")
    table(s, rx, 1.72, rw, [["Spread", "Sharpe", "Max drawdown"],
                            ["3-2-1 crack", "0.36 → 0.32", "−$159k → −$156k"],
                            ["Brent − WTI", "−0.55 → −0.47", "−$321k → −$307k"],
                            ["HO − RB", "−0.07 → +0.02", "**−$356k → −$182k**"],
                            ["Crack, Johansen", "−0.27 → −0.17", "−$224k → −$159k"]], [1.3, 1.2, 1.7], size=11.5, row_h=0.31)
    textbox(s, rx, 3.45, rw, 2.5, [
        "**The rule:** units = $6,000 ÷ (σ̂ × $ per unit).",
        "After 24 Feb 2022 GARCH halves the units in **6 trading days**; the frozen sd takes **89**.",
        "Risk above twice the budget: **0.4–0.9%** of days with GARCH, **5.7–6.3%** with the frozen sd.",
        "Today σ̂ is at the 96th, 57th and 97th percentile of its history, so the rule allows 1.0, 16.2 and 1.4 units (crack, Brent − WTI, HO − RB)."],
        size=12.5)
    callout(s, "risk control, not a signal: GARCH halves HO − RB's drawdown, and cannot turn a losing signal into a winner.")

    # ---- Step 9 -------------------------------------------------------------------------------------------
    s = new_slide("Step 9: when a gap opens, which leg closes it? (VECM)", section("## Step 9: when the gap opens"))
    b = picture(s, FIG / "21_vecm_who_moves.png", X0, 1.35, W,
                "Schematic of a VECM; t-statistics of each leg's error-correction term; share of the gap each side closed within 63 days.",
                max_h=3.4)
    textbox(s, X0, b + 0.1, W, 0.7, [
        "**In plain English.** Two dogs on one leash: when it goes tight, one steps back. α is the share of yesterday's gap a leg closes today. "
        "**Crack:** crude closes ~53% of a gap in 63 days while the products move ~30% further away. **HO − RB:** ULSD closes ~27%.",
        "**Brent − WTI:** no clear leader. Rolling fits agree: crude is the significant corrector in 16% (crack) and 29% (Brent − WTI) of windows, diesel in 24% of HO − RB."],
        size=12, space_after=3)
    callout(s, "trade the spread as one unit, not one leg. Who moves is clear only for the crack, and the pull is slow.")

    # ---- Step 9b ------------------------------------------------------------------------------------------
    s = new_slide("Step 9b: at the average exit, most of the gap is still open", section("## Step 9b: what the VECM means"))
    b = picture(s, FIG / "22_vecm_gap_decay.png", X0, 1.35, 8.1,
                "Share of a gap still open by days held: the VECM curve, the measured local projection with 95 percent bands, and the GARCH volatility decay.")
    textbox(s, X0 + 8.35, 1.35, W - 8.35, 2.6, [
        "**Measured:** the held-contract return over the next h days regressed on today's gap. Closure at 63 days is significant for all three (t = 2.2, 2.6, 3.1).",
        "At the average exit (23–31 days), **80–90%** of the gap is still open.",
        "Volatility halves in 26–38 days; the gap in 70–96."], size=12, space_after=4)
    table(s, X0, 4.15, W, [["VECM / measured", "Gap half-life", "Still open at exit", "After 3 months", "After 1 year"],
                          ["3-2-1 crack", "70 days", "74% / 90%", "54% / 77%", "8% / 19%"],
                          ["Brent / WTI", "77 days", "81% / 90%", "57% / 74%", "10% / **57%**"],
                          ["HO / RB", "96 days", "83% / 80%", "64% / 62%", "16% / 8%"]], [1.6, 1.2, 1.5, 1.5, 1.4], size=12, row_h=0.34)
    callout(s, "daily mean reversion exits too early. Volatility fades faster than the gap, so size by σ̂, not by how wide the gap is.")

    # ---- Appendix: GARCH in full --------------------------------------------------------------------------
    s = new_slide("Test 6b: GARCH in full, parameters and checks", section("## Test 6b: GARCH in full"))
    caption(s, X0, 1.35, W, "Median fit by era (rolling 756 days, refit monthly): a / b / ν")
    table(s, X0, 1.7, W, [["Era", "3-2-1 crack", "Brent − WTI", "HO − RB", "Vol-shock half-life (crack / B − W / HO − RB)"],
                          ["2007–13", "0.14 / 0.85 / 113", "0.15 / 0.84 / 10", "0.11 / 0.87 / 26", "99 / 48 / 40 days"],
                          ["2014–19", "0.09 / 0.83 / 18", "0.08 / 0.88 / 21", "0.08 / 0.89 / 14", "9 / 16 / 20 days"],
                          ["2020–26", "0.11 / 0.87 / 10", "0.15 / 0.83 / 10", "0.10 / 0.88 / 12", "26 / 36 / 38 days"]],
          [0.9, 1.7, 1.7, 1.7, 3.1], size=12, row_h=0.34)
    caption(s, X0, 3.25, W, "Each day's move divided by the forecast")
    table(s, X0, 3.6, W, [["Spread", "Kurtosis: window sd → GARCH", "Lag-1 ACF of squared move", "QLIKE gain (t)"],
                          ["3-2-1 crack", "13.4 → 5.4", "0.31 → 0.04", "−0.43 (−4.9)"],
                          ["Brent − WTI", "101.6 → 6.2", "0.19 → 0.04", "−1.15 (−2.6)"],
                          ["HO − RB", "79.2 → 5.9", "0.39 → 0.11", "−0.76 (−2.7)"]],
          [1.2, 2.3, 2.3, 1.6], size=12, row_h=0.34)
    textbox(s, X0, 5.1, W, 0.9, [
        "a + b near 1 means storms persist; ν of 10–26 means fat tails. Half-life = ln 2 ÷ (−ln(a + b)).",
        "The fit converges in 86%, 78% and 84% of months; otherwise σ̂ falls back to the window sd. Clustering is reduced, not removed: Ljung–Box on squared moves still rejects."],
        size=12, space_after=3)
    callout(s, "σ̂ is a good, not perfect, size forecast: it removes most of the fat tail and clustering that a frozen sd leaves in.")

    # ---- Appendix: GARCH forecast -------------------------------------------------------------------------
    s = new_slide("Test 6c: GARCH forecast against realised volatility", section("## Test 6: can't predict direction"))
    b = picture(s, FIG / "23_garch_forecast_vs_realised.png", X0, 1.35, W,
                "GARCH forecast, frozen window sd and realised volatility for the three spreads, 2020 to 2022.", max_h=3.3)
    textbox(s, X0, b + 0.15, W, 1.0, [
        "In 2020 and 2022 the GARCH forecast tracks realised volatility within days; the frozen sd steps up a month at a time.",
        "Brent − WTI's flat stretch in 2020–21 is a run of months where the fit failed and σ̂ fell back to a constant.",
        "Same gated trades, only the units swapped (gross): HO − RB's drawdown falls from −$330k to −$157k; the crack's rises slightly (−$139k to −$158k)."],
        size=12.5, space_after=4)
    callout(s, "size on σ̂, not on a frozen window: it reacts in days, not months.")

    # ---- Appendix: VECM in full ---------------------------------------------------------------------------
    s = new_slide("Test 7b: VECM in full, three specifications", section("## Test 7b: VECM in full"))
    caption(s, X0, 1.35, 7.6, "Known vector: logs (full sample) and dollars (rolling 3 years)")
    table(s, X0, 1.7, 7.6, [["Spread: leg", "Logs: α a day (t)", "Rolling: corrects in"],
                            ["Crack: WTI", "**+0.017 (2.6)**", "**16%**"],
                            ["Crack: RBOB", "+0.007 (1.6), wrong sign", "0.5%"],
                            ["Crack: ULSD", "+0.007 (1.8), wrong sign", "2%"],
                            ["Brent / WTI: WTI", "+0.012 (1.3)", "**29%**"],
                            ["Brent / WTI: Brent", "+0.003 (0.4), wrong sign", "5%"],
                            ["HO / RB: ULSD", "−0.007 (−1.5)", "**24%**"],
                            ["HO / RB: RBOB", "−0.000 (−0.0), wrong sign", "9%"],
                            ["Gap on itself", "−0.010 / −0.009 / −0.007", "half-lives 70 / 77 / 96 d"]],
          [2.0, 2.7, 2.3], size=11.5, row_h=0.3)
    rx, rw = X0 + 7.9, W - 7.9
    caption(s, rx, 1.35, rw, "Unrestricted Johansen, dollars: α (t)")
    table(s, rx, 1.7, rw, [["Leg", "Vector 1", "Vector 2"],
                           ["WTI", "−0.0008 (−1.2)", "−0.0017 (−1.1)"],
                           ["RBOB", "**−0.0022 (−2.9)**", "−0.0046 (−2.6)"],
                           ["ULSD", "−0.0008 (−1.0)", "−0.0007 (−0.4)"],
                           ["Vector", "CL − 1.74 HO", "RB − 0.05 HO"]], [0.8, 1.6, 1.6], size=11, row_h=0.3)
    textbox(s, X0, 4.6, W, 1.4, [
        "Gap = ln A − ln B: the first leg corrects with α < 0, the second with α > 0.",
        "The unrestricted VECM picks RBOB, but its fitted vectors are not the traded crack.",
        "Local projection, 63 days: crack crude +53% (±53), products −30% (±45); ULSD in HO − RB +27% (±33)."],
        size=12.5, space_after=4)
    callout(s, "the correcting leg depends on the specification, so trade the spread as one unit.")

    # ---- Appendix: VECM rolling ---------------------------------------------------------------------------
    s = new_slide("Test 7c: rolling VECM loadings on the crack", section("## Test 7: when the gap is off"))
    picture(s, FIG / "24_vecm_rolling.png", X0, 1.35, 7.4,
            "Rolling t-statistics of each crack leg's VECM loading with the vector fixed at 3:2:1, and the Johansen variant's weights.", max_h=4.55)
    textbox(s, X0 + 7.65, 1.35, W - 7.65, 4.6, [
        "**Top:** t-statistic of each leg's loading, rolling 3 years. CL corrects when t > +2; RB and HO when t < −2.",
        "CL crosses +2 in **16%** of windows; RB and HO cross −2 in 0.5% and 2%.",
        "In 2011, 2022 and 2026 all three legs jump together: a common shock, not a correction.",
        "**Bottom:** the free Johansen vector is usable in only 30% of months, and its weights jump."],
        size=12.5, space_after=6)
    callout(s, "one more reason not to trade a single leg: the leader changes, and the free vector is unstable.")

    p.save(str(OUT))
    print("wrote", OUT.relative_to(ROOT), "with", len(p.slides), "slides")


if __name__ == "__main__":
    build()
