"""Fetch the live site and check every link on the front page actually resolves.

GitHub answers 503 when a burst of requests hits it, which once looked like eight broken records (2026-10-13), so each link
is retried with a pause and only a link that fails every attempt counts as broken. Usage: python3 CANON/live_links.py [--check]
"""
import re, sys, time, urllib.request, urllib.error
U = "https://hmahaffeyges.github.io/IAM-Validation/"
SECTIONS = ["Start here: the daily log", "Recent Accomplishments", "Every test, numbered", "The road ahead", "How we test the physics", "Who is doing this, and why"]
def get(u, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(u, timeout=120) as r: return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == tries - 1: return e.code, ""
        except Exception:
            if i == tries - 1: return "ERR", ""
        time.sleep(4 * (i + 1))
    return "ERR", ""
def main():
    code, s = get(U + "?v=livecheck")
    P = [] if code == 200 else [f"the front page answered {code}"]
    for k in SECTIONS:
        if k not in s: P.append(f"section missing from the live page: {k}")
    if "MANIFEST.md" not in s: P.append("the live page does not link MANIFEST.md (the Master Testing Manifest)")
    for k in ("TODO.md", "ROADMAP.md", "next_jobs.json"):
        if k in s: P.append(f"the live page links the retired list {k}")
    links = sorted({l for l in re.findall(r'(?:href|src)="([^"#]+)"', s) if not l.startswith("mailto:")})
    bad = []
    for l in links:
        if "hypothes.is" in l: continue
        c, _ = get(l if l.startswith("http") else U + l)
        if c != 200: bad.append((c, l))
        time.sleep(0.3)
    for pth in ("read/", "book/", "pdf/IAMs_Law_and_Order.pdf", "epub/IAMs_Law_and_Order.epub", "go/"):
        c, _ = get(U + pth)
        if c != 200: bad.append((c, pth))
    if bad: P.append(f"{len(bad)} link(s) did not resolve after retries: " + ", ".join(f"{c} {l}" for c, l in bad[:6]))
    print(f"live_links: {'ok' if not P else 'REFUSED: ' + '; '.join(P)} ({len(links)} links on the page)")
    return 1 if P else 0
if __name__ == "__main__": sys.exit(main())
