#!/usr/bin/env python3
"""Reading themes of the website: Paper (default), Dark, Sepia. Writes website/static/themes.css and website/build/CONTRAST.md.

  python3 website/build/themes.py          (run by build.sh; exits 1 if any text colour misses WCAG AA)

Every text colour is checked against every background it is drawn on, at the WCAG 2.x AA threshold for normal text (4.5:1). A colour
that misses it is moved in lightness only (darker on a light page, lighter on a dark one) until it passes; the report lists the
contrast of every pair as shipped. Body text, background and the Dark theme's colours are the author's: Paper #2B2B2B on #FAF7F0,
Dark #E6E3DC on #1E1F22. Figures are never recoloured: in Dark and Sepia they sit on a light rounded panel (site.css).
"""
import colorsys, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE.parent / "static"

THEMES = {
    "paper": dict(bg="#FAF7F0", fg="#2B2B2B", panel="#F1ECE1", code="#ECE6D9", muted="#5A5751", link="#1F4F8F", rule="#D8D0C0",
                  figpanel="#FFFFFF",
                  der="#1F5FA8", cal="#2E7D4F", fit="#8A6508", con="#7B3A9A", pre="#B0352A", obs="#555555",
                  st_g="#C8E6C9", st_b="#BBDEFB", st_a="#FFE0B2", st_r="#FFCDD2", st_fg="#2B2B2B",
                  acc12="#B5452F", acc3="#A3530F", acc45="#1C7391", acc6="#2F5FB8", acc7="#774BAF", on_acc="#FFFFFF"),
    "dark": dict(bg="#1E1F22", fg="#E6E3DC", panel="#26282C", code="#2C2E33", muted="#B5B1A8", link="#8DB8F2", rule="#3C3E44",
                 figpanel="#F4F1EA",
                 der="#7FB0F0", cal="#6CC995", fit="#E3BF55", con="#C99AE6", pre="#F08A7C", obs="#BDBDBD",
                 st_g="#2F5A33", st_b="#27476B", st_a="#6B4A1E", st_r="#6B2A2F", st_fg="#F2F2F2",
                 acc12="#EF8A6F", acc3="#F3A85A", acc45="#5CC4E0", acc6="#7AA2F0", acc7="#B79AF0", on_acc="#1E1F22"),
    "sepia": dict(bg="#F4ECD8", fg="#3B2F24", panel="#EADFC5", code="#E4D7B9", muted="#5C4C38", link="#2A4E80", rule="#D2C29E",
                  figpanel="#FFFDF7",
                  der="#1F5FA8", cal="#2E7D4F", fit="#7E5C06", con="#7B3A9A", pre="#A8322A", obs="#555555",
                  st_g="#C8E6C9", st_b="#BBDEFB", st_a="#FFE0B2", st_r="#FFCDD2", st_fg="#3B2F24",
                  acc12="#A8402B", acc3="#985010", acc45="#1A6A86", acc6="#2C58AB", acc7="#6E45A3", on_acc="#FFFFFF"),
}
FIXED = {"bg", "fg"}          # the author's colours: checked, never moved

# (text token, [background tokens]) drawn as text at normal size
PAIRS = [("fg", ["bg", "panel", "code"]), ("muted", ["bg", "panel"]), ("link", ["bg", "panel", "code"]),
         ("der", ["bg", "panel"]), ("cal", ["bg", "panel", "code"]), ("fit", ["bg", "panel"]), ("con", ["bg", "panel"]),
         ("pre", ["bg", "panel", "code"]), ("obs", ["bg", "panel"]),
         ("st_fg", ["st_g", "st_b", "st_a", "st_r"]),
         ("acc12", ["bg", "panel", "on_acc"]), ("acc3", ["bg", "panel", "on_acc"]), ("acc45", ["bg", "panel", "on_acc"]),
         ("acc6", ["bg", "panel", "on_acc"]), ("acc7", ["bg", "panel", "on_acc"])]
AA = 4.5


def lum(h):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def shift(h, dl):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hh, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1, max(0, l + dl))
    r, g, b = colorsys.hls_to_rgb(hh, l, s)
    return "#%02X%02X%02X" % tuple(round(x * 255) for x in (r, g, b))


def settle(name, t):
    moved = []
    dark = lum(t["bg"]) < 0.2
    for tok, bgs in PAIRS:
        if tok in FIXED:
            continue
        for _ in range(200):
            if min(ratio(t[tok], t[b]) for b in bgs) >= AA:
                break
            t[tok] = shift(t[tok], 0.01 if dark else -0.01)
        else:
            raise SystemExit(f"{name}: cannot reach AA for {tok}")
    return t


def css(themes):
    out = ["/* Reading themes, written by website/build/themes.py (WCAG AA checked; see website/build/CONTRAST.md). */"]
    def block(sel, t):
        v = [f"--bg: {t['bg']}", f"--fg: {t['fg']}", f"--panel: {t['panel']}", f"--code-bg: {t['code']}", f"--muted: {t['muted']}",
             f"--link: {t['link']}", f"--rule: {t['rule']}", f"--figpanel: {t['figpanel']}",
             f"--chip-der: {t['der']}", f"--chip-cal: {t['cal']}", f"--chip-fit: {t['fit']}", f"--chip-con: {t['con']}",
             f"--chip-pre: {t['pre']}", f"--chip-obs: {t['obs']}",
             f"--st-g: {t['st_g']}", f"--st-b: {t['st_b']}", f"--st-a: {t['st_a']}", f"--st-r: {t['st_r']}", f"--st-fg: {t['st_fg']}",
             f"--on-accent: {t['on_acc']}", f"--accent: {t['acc6']}"]
        out.append(sel + " {\n  " + ";\n  ".join(v) + ";\n}")
        for parts, k in (("1, 2", "acc12"), ("3", "acc3"), ("4, 5", "acc45"), ("6", "acc6"), ("7", "acc7")):
            sels = ", ".join(f"{sel} body.part-{p.strip()}" for p in parts.split(","))
            out.append(f"{sels} {{ --accent: {t[k]}; }}")
    block(':root, html[data-theme="paper"]', themes["paper"])
    block('html[data-theme="dark"]', themes["dark"])
    block('html[data-theme="sepia"]', themes["sepia"])
    out.append('html[data-theme="dark"] { color-scheme: dark; } html[data-theme="paper"], html[data-theme="sepia"] { color-scheme: light; }')
    return "\n".join(out) + "\n"


def main():
    themes = {k: settle(k, dict(v)) for k, v in THEMES.items()}
    rep = ["# Reading themes: contrast (WCAG 2.x AA, normal text, threshold 4.5:1)", "",
           "Written by `website/build/themes.py` at each build. Every text colour against every background it is drawn on.", ""]
    worst, bad = 99, []
    for name, t in themes.items():
        rep += [f"## {name.capitalize()}", "", "| text | on | colours | contrast |", "|---|---|---|---:|"]
        for tok, bgs in PAIRS:
            for b in bgs:
                r = ratio(t[tok], t[b])
                worst = min(worst, r)
                if r < AA:
                    bad.append((name, tok, b, r))
                rep.append(f"| {tok} | {b} | `{t[tok]}` on `{t[b]}` | {r:.2f} |")
        changed = [k for k in t if t[k] != THEMES[name][k]]
        rep += ["", "Adjusted from the starting palette to reach AA: " + (", ".join(f"{k} {THEMES[name][k]} -> {t[k]}" for k in changed) or "none") + ".", ""]
    rep.insert(4, f"Lowest contrast of any pair: {worst:.2f}:1. Pairs below 4.5:1: {len(bad)}.\n")
    (HERE / "CONTRAST.md").write_text("\n".join(rep) + "\n")
    (STATIC / "themes.css").write_text(css(themes))
    print(f"themes: paper, dark, sepia; lowest contrast {worst:.2f}:1; below AA: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
