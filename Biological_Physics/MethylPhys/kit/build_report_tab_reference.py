#!/usr/bin/env python3
"""Describe the report the chain produces - every tab - from a real report, never from memory.

Author's instruction 2026-09-25: the SOP and Issue 003 "should have every tab described and explained and the
CMB pass/fails etc. It should have screenshots of the html for each tab too. Its literally the operating
manual."

WHY THIS IS A GENERATOR AND NOT PROSE. A tab list written by hand goes stale the first time a tab is added -
which has happened twice this month (Troubleshooting, Red flags). This reads a finished report, enumerates
what is actually in it, and writes three outputs that the SOP, the manual and the reviewer manifest all
consume:

    doors/REPORT_TAB_REFERENCE.md      the tab-by-tab reference, for a reader of the repository
    manual/report_tabs.json            the same content as data, for the Issue 003 build
    manual/report_tab_figures/*.png    one figure per tab

ON THE FIGURES, STATED PLAINLY. These are NOT browser screenshots. A headless browser cannot be installed in
this environment (Quick Look is sandboxed off; the Playwright download resolves to a denylisted host), so each
figure is RENDERED from that tab's own HTML - its headings, its prose and its tables, in the report's own dark
palette. Every figure says so in its caption. If a real browser screenshot is wanted, drop a PNG named
<tab>.png into manual/report_screenshots/ and this script prefers it, captioning it as a screenshot; nothing
else needs changing.
"""
import html as _html
import json
import os
import re
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.dirname(HERE)
OUT_MD = os.path.join(MP, "doors", "REPORT_TAB_REFERENCE.md")
OUT_JSON = os.path.join(MP, "manual", "report_tabs.json")
FIGDIR = os.path.join(MP, "manual", "report_tab_figures")
SHOTDIR = os.path.join(MP, "manual", "report_screenshots")

# What each tab is FOR. The only hand-written content in this file: a purpose cannot be derived from HTML.
# A tab that appears in a report and is missing from here is reported as undocumented rather than skipped.
PURPOSE = {
 "reading":    ("SPECIMEN", "The reading itself: what is in the sample and how each cell reads - every present cell's A = H/H_min "
                            "on its own identity loci against the fixed point 1.00, with its tier word; any foreign cell detected; "
                            "and the instrument that took the reading. If a clinician reads one tab, it is this one."),
 "howto":      ("REFERENCE", "How to read the gauge: what A measures, what each tier means, and what A = 1.00 is and is not."),
 "cells":      ("SPECIMEN", "Every atlas cell, scored or not: each cell's A on its own identity loci with its interval and how many "
                            "of its loci the array read, then the trace detection of anything non-blood, and what limits a "
                            "claim about one cell."),
 "sky":        ("SPECIMEN", "The sky: a Mollweide plate of this specimen's own residuals - its methylation at each address minus "
                            "what its own composition predicts - with the statistics behind the plate. Compared to no picture "
                            "of anyone else."),
 "physics":    ("REFERENCE", "The physics under the gauge: Landauer's bound, the entropy floor H_min, and why the zero is a fixed "
                            "point rather than a group of people."),
 "story":      ("REFERENCE", "What astro-genetics is, in the author's words, with a link to the paper."),
 "reference":  ("REFERENCE", "The instrument: its physical constants (the eight floors - a cell's class names the floor it is "
                            "divided by, nothing else) and what a laboratory's array needs before a reading is taken."),
 "coverage":   ("REFERENCE", "What is lit and what is reserved: the five substrates, the specimen types, and what lighting one "
                            "cell requires."),
 "flags":      ("SPECIMEN", "Red flags: everything this run refused, withheld or could not measure, in one place, ordered by "
                            "severity - STOP, WITHHELD, CAUTION, NOTE."),
 "safeguards": ("SPECIMEN", "Every guard and whether it passed on this specimen, with the Stage 0 custody record."),
 "trouble":    ("REFERENCE", "Troubleshooting: every refusal the chain can print, what it means, and what the operator does."),
 "integrity":  ("SPECIMEN", "The fail-safes that kept this reading honest: what this run refused and why, each safeguard with its "
                            "cosmology twin, the instrument constants this run read, and every file it read with its hash."),
 "chain":      ("REFERENCE", "The chain that produced the reading, stage by stage, derived from the code rather than described - "
                             "so a stage cannot be claimed that the code does not call."),
 "files":      ("REFERENCE", "Every file the chain uses, enumerated from the live tree with its role and SHA-256."),
 "findings":   ("REFERENCE", "Findings that changed a reported number, with the procedure that sealed each."),
 "record":     ("REFERENCE", "The validation record: every series and sealed procedure, linked."),
 "run":        ("SPECIMEN", "Run it yourself: what produced this reading (chain commit, a hash of every input) and how to clone, "
                            "verify and run the chain."),
}

LABELS = {"reading": "Reading", "cells": "Cells", "sky": "Sky", "flags": "Red flags",
          "safeguards": "Safeguards", "integrity": "Integrity", "run": "Run", "coverage": "Coverage",
          "chain": "Chain", "files": "Files", "howto": "How to read", "story": "Story", "physics": "Physics",
          "reference": "Reference", "record": "Record", "findings": "Findings", "trouble": "Troubleshooting",
          "roadmap": "Roadmap"}


def strip(s):
    """Text as a reader sees it: tags removed AND entities resolved.

    2026-09-25: the first version left entities raw, so a figure showed "&Delta;G ATP &divide; (R &middot; T)"
    where the report shows "\u0394G_ATP / (R\u00b7T)". A figure of a page must read like the page.
    """
    t = re.sub(r"<(br|/tr)[^>]*>", " \u00b7 ", s)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", _html.unescape(t)).replace("\u00a0", " ").strip()


def tables_of(sec, limit=14):
    """Each table as (header, rows) of plain cells - the registry and the reading live in these."""
    out = []
    for m in re.finditer(r"<table[^>]*>(.*?)</table>", sec, flags=re.S):
        body = m.group(1)
        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.S):
            cells = [strip(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S)]
            if any(cells):
                rows.append(cells)
        if rows:
            out.append(rows[:limit])
    return out


def tabs_of(path):
    t = open(path, encoding="utf-8").read()
    order = [m.group(1) for m in re.finditer(r"data-t='(\w+)'", t)]
    secs = dict(re.findall(r"<section class='tab[^']*' id='(\w+)'>(.*?)</section>", t, flags=re.S))
    seen, ordered = set(), []
    for k in order + sorted(secs):
        if k in secs and k not in seen:
            seen.add(k)
            ordered.append(k)
    return t, ordered, secs


def headings(sec):
    return [strip(m.group(2)) for m in re.finditer(r"<h([23])[^>]*>(.*?)</h\1>", sec, flags=re.S)]


def cmb_registry(sec):
    rows = re.findall(r"<tr><td class='m'><b>(PASS|FAIL|NOT_RUN|NOT_APPLICABLE|NOT_BUILT)</b></td>"
                      r"<td>([^<]+)</td>", sec)
    return [{"state": s, "tool": _html.unescape(n).strip()} for s, n in rows]


def figure(tab, sec, path):
    """Render a tab as a figure in the report's own palette. Not a screenshot - see the module docstring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, FG, ACC, DIM = "#0b0d12", "#e8e8ea", "#8ab4f8", "#9aa0a6"
    fig = plt.figure(figsize=(9.5, 6.4), dpi=170)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")
    y = 0.955
    ax.text(0.035, y, LABELS.get(tab, tab).upper(), color=ACC, fontsize=15, weight="bold",
            family="DejaVu Sans", va="top")
    kind = PURPOSE.get(tab, ("", ""))[0]
    ax.text(0.965, y, kind, color=DIM, fontsize=9, ha="right", va="top", family="DejaVu Sans")
    y -= 0.075
    blocks = re.findall(r"<(h3|p|td|th)[^>]*>(.*?)</\1>", sec, flags=re.S)
    shown = 0
    for kindtag, body in blocks:
        txt = strip(body)
        if not txt or len(txt) < 3:
            continue
        if kindtag == "h3":
            if y < 0.12:
                break
            y -= 0.012
            ax.text(0.035, y, txt[:96], color=ACC, fontsize=10.5, weight="bold", va="top",
                    family="DejaVu Sans")
            y -= 0.045
            shown += 1
        elif kindtag == "p":
            wrapped = textwrap.fill(txt, 112)[:1500]
            n = wrapped.count("\n") + 1
            if y - 0.028 * n < 0.10:
                break
            ax.text(0.035, y, wrapped, color=FG, fontsize=8.2, va="top", family="DejaVu Sans",
                    linespacing=1.45)
            y -= 0.0255 * n + 0.016
            shown += 1
        if shown > 14:
            break
    tbls = tables_of(sec, limit=20)
    for rows in tbls[:2]:
        if y < 0.20:
            break
        y -= 0.014
        ncol = max(len(r) for r in rows)
        width = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(ncol)]
        width = [min(w, 46) for w in width]
        fs = 7.0 if sum(width) < 150 else 5.6
        for j, r in enumerate(rows):
            if y < 0.115:   # keep clear of the caption
                ax.text(0.035, y, "... table continues in the report", color=DIM, fontsize=6.4, va="top",
                        family="DejaVu Sans Mono")
                break
            line = "  ".join((r[i] if i < len(r) else "")[:width[i]].ljust(width[i]) for i in range(ncol))
            ax.text(0.035, y, line[:210], color=(ACC if j == 0 else FG), fontsize=fs, va="top",
                    family="DejaVu Sans Mono",
                    weight=("bold" if j == 0 else "normal"))
            y -= 0.0235 if fs > 6 else 0.0195
        y -= 0.02
    ax.text(0.035, 0.045, "Rendered from this tab's own HTML (not a browser screenshot) - "
                          "build_report_tab_reference.py", color=DIM, fontsize=6.6, va="top",
            family="DejaVu Sans")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main(report=None):
    if report is None:
        cands = [os.path.join(MP, "chain", "example_runs", d, f)
                 for d in sorted(os.listdir(os.path.join(MP, "chain", "example_runs")))
                 for f in (sorted(os.listdir(os.path.join(MP, "chain", "example_runs", d)))
                           if os.path.isdir(os.path.join(MP, "chain", "example_runs", d)) else [])
                 if f.endswith(".html")]
        assert cands, "no example report to read - run the chain first"
        report = cands[-1]
    t, order, secs = tabs_of(report)
    os.makedirs(FIGDIR, exist_ok=True)
    commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    rows, undocumented = [], []
    for tab in order:
        sec = secs[tab]
        if tab not in PURPOSE:
            undocumented.append(tab)
        kind, why = PURPOSE.get(tab, ("UNDOCUMENTED", "This tab is in the report and has no description "
                                                      "here. Add one to PURPOSE."))
        shot = os.path.join(SHOTDIR, tab + ".png")
        if os.path.exists(shot):
            fig_rel, is_shot = os.path.relpath(shot, os.path.dirname(OUT_MD)), True
        else:
            fp = os.path.join(FIGDIR, tab + ".png")
            figure(tab, sec, fp)
            fig_rel, is_shot = os.path.relpath(fp, os.path.dirname(OUT_MD)), False
        rows.append({"tab": tab, "label": LABELS.get(tab, tab), "kind": kind, "purpose": why,
                     "kb": len(sec) // 1000, "tables": sec.count("<table"), "rows": sec.count("<tr"),
                     "headings": headings(sec)[:14],
                     "opens": strip(sec)[:240],
                     "figure": fig_rel.replace(" ", "%20"), "is_screenshot": is_shot,
                     "cmb": cmb_registry(sec) if tab == "safeguards" else []})
    reg = next((r["cmb"] for r in rows if r["cmb"]), [])
    meta = {"generated_from": os.path.relpath(report, MP), "commit": commit,
            "n_tabs": len(rows), "report_mb": round(len(t) / 1e6, 2),
            "n_specimen": sum(1 for r in rows if r["kind"] == "SPECIMEN"),
            "n_reference": sum(1 for r in rows if r["kind"] == "REFERENCE"),
            "figures_are_screenshots": any(r["is_screenshot"] for r in rows)}
    json.dump({"_meta": meta, "tabs": rows, "cmb_registry": reg}, open(OUT_JSON, "w"), indent=1)

    L = [f"# The report, tab by tab - the operating reference",
         "",
         f"**Generated** by [`build_report_tab_reference.py`](../kit/build_report_tab_reference.py) from "
         f"`{meta['generated_from']}` at commit `{commit}`. Not written by hand: a tab list written by hand "
         f"goes stale the first time a tab is added, which happened twice in September. Re-run it after any "
         f"change to the report builder.", "",
         f"One run produces **one self-contained HTML file of {meta['report_mb']} MB with "
         f"{meta['n_tabs']} tabs** - {meta['n_specimen']} carrying this specimen's own measurements and "
         f"{meta['n_reference']} carrying reference material that is identical in every report. The "
         f"distinction matters: a reference tab tells you how the instrument works, and only a specimen tab "
         f"tells you anything about the patient.", "",
         "## The figures", "",
         ("These are **browser screenshots**, supplied in `manual/report_screenshots/`."
          if meta["figures_are_screenshots"] else
          "These are **not browser screenshots.** A headless browser cannot be installed in the build "
          "environment (Quick Look is sandboxed off and the Playwright download resolves to a denylisted "
          "host), so each figure is rendered from that tab's own HTML - its headings and prose, in the "
          "report's palette - and says so in its caption. To replace one with a real screenshot, put a PNG "
          "named `<tab>.png` in `manual/report_screenshots/` and re-run this script; it prefers it."), "",
         "## Every tab", ""]
    for r in rows:
        L += [f"### {r['label']}  ·  `{r['tab']}`  ·  {r['kind']}", "",
              r["purpose"], "",
              f"![{r['label']} tab]({r['figure']})", "",
              f"*{r['kb']} KB, {r['tables']} tables, {r['rows']} rows.*  "
              f"Sections: {', '.join('**' + h + '**' for h in r['headings'][:8]) if r['headings'] else 'none'}",
              ""]
    if reg:
        L += ["## The CMB tool register, as it reads on this specimen", "",
              "Every method borrowed from CMB analysis, with a check that runs on the finished bundle. A FAIL "
              "is also emitted to Red flags as `CMB_TOOL_FAIL`. **NOT_BUILT entries are listed on purpose** - "
              "the shelf is part of the record, and [`ENHANCEMENTS.md`](ENHANCEMENTS.md) ranks them.", "",
              "| state | method |", "|---|---|"]
        for c in sorted(reg, key=lambda c: (c["state"] != "FAIL", c["state"], c["tool"])):
            L.append(f"| `{c['state']}` | {c['tool']} |")
        L += ["", "To add a tool: append one entry to `TOOLS` in "
                  "[`cmb_tools.py`](../chain/cmb_tools.py) with a `check(bundle)`. The table, the counts and "
                  "the red-flag routing all follow.", ""]
    if undocumented:
        L += ["## Undocumented tabs", "",
              "These are in the report with no description in the generator. **Fix by adding them to "
              "`PURPOSE`** - they are listed rather than skipped so the omission cannot hide:", ""] + \
             [f"- `{u}`" for u in undocumented] + [""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))
    print(f"REPORT_TAB_REFERENCE.md: {len(rows)} tabs ({meta['n_specimen']} specimen, "
          f"{meta['n_reference']} reference), {len(reg)} CMB tools, "
          f"{'screenshots' if meta['figures_are_screenshots'] else 'rendered figures'}")
    if undocumented:
        print("  UNDOCUMENTED TABS:", undocumented)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
