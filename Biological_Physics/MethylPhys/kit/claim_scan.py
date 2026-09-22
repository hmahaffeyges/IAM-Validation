#!/usr/bin/env python3
"""claim_scan.py - the claims in claims.json, checked against the RENDERED document.

    python3 claim_scan.py ../manual/IAMPerformance_GAPEIssue003_RC1.pdf

Exit 0 when no forbidden claim is asserted and every required passage is present; exit 1 otherwise, naming the
page and the sentence. Written 2026-09-22 after three defects were found this way and none by grepping the source.

A hit is cleared only when the correction appears in the NEIGHBOURING sentences, because a correction placed after
a long block is not a correction of the block - an appended glossary note left the wrong entry standing on its own
page. The window is what makes that detectable.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 2


def pages_of(path):
    import pypdfium2 as pdfium
    d = pdfium.PdfDocument(path)
    return [re.sub(r"\s+", " ", d[i].get_textpage().get_text_range()) for i in range(len(d))]


def main():
    if len(sys.argv) < 2:
        print("claim_scan: no document given"); return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"claim_scan: SKIPPED - {os.path.basename(path)} not present"); return 0
    spec = json.load(open(os.path.join(HERE, "claims.json"), encoding="utf-8"))
    pages = pages_of(path)
    failures = []
    for rule in spec.get("must_not_assert", []):
        for i, text in enumerate(pages):
            sents = re.split(r"(?<=[.!?]) ", text)
            for k, s in enumerate(sents):
                if not any(t in s for t in rule["terms"]): continue
                if not any(c in s for c in rule["contradict"]): continue
                near = " ".join(sents[max(0, k - WINDOW):k + WINDOW + 1])
                if any(e in near for e in rule.get("exculpate", [])): continue
                failures.append(f"[{rule['id']}] p{i+1}: {s.strip()[:160]}")
    for rule in spec.get("must_contain", []):
        n = sum(1 for t in pages if rule["text"] in t)
        if n < rule.get("at_least", 1):
            failures.append(f"[{rule['id']}] present on {n} pages, requires {rule.get('at_least',1)}: "
                            f"\"{rule['text'][:60]}\" - {rule['why']}")
    print(f"claim_scan: {len(pages)} pages, {len(spec.get('must_not_assert',[]))} forbidden claims and "
          f"{len(spec.get('must_contain',[]))} required passages checked")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures[:30]: print("   " + f)
        return 1
    print("claim_scan: PASS - no forbidden claim asserted, every required passage present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
