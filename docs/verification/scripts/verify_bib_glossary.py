#!/usr/bin/env python3
"""verify_bib_glossary.py -- checks for the bibliography audit and the regenerated glossary (2026-10-03, base 5f5997b).

Run from the repo root:  python3 docs/verification/scripts/verify_bib_glossary.py [--offline]
 1. every DOI in docs/book/bib_additions.bib and docs/book/corrected_entries.bib resolves on api.crossref.org and its
    CrossRef title shares >= 60 % of its words with the entry title (10.48550/arXiv.* DOIs are checked on the arXiv API);
 2. every insertion block in docs/book/bib_insertions.json: the anchor line occurs exactly once in its file, the old text
    occurs once in that line, and every cited key is in iam.bib or bib_additions.bib;
 3. docs/book/appendices/app_F_glossary.tex: braces and $ balance per entry, every \\ref resolves to a \\label in the book,
    every \\cite key is in iam.bib, every number of two or more digits occurs in the chapter/appendix text,
    and no retired pattern of CANON/iam_canon.json (or the book's retired list) occurs.
"""
import json, re, sys, os, unicodedata, urllib.request, urllib.parse, time
ROOT=os.getcwd(); B=os.path.join(ROOT,'docs','book')
OFF='--offline' in sys.argv
def rd(p): return open(p,encoding='utf-8',errors='replace').read()
def keys_of(t): return re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,',t)
def norm(s): return re.sub(r'[^a-z0-9 ]',' ',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())
out=[]; fail=0
iam=rd(os.path.join(B,'iam.bib')); add=rd(os.path.join(B,'bib_additions.bib')); cor=rd(os.path.join(B,'corrected_entries.bib'))
K=set(keys_of(iam))|set(keys_of(add))
def entries(t):
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}',t,re.S):
        body=m.group(3); d=re.search(r'doi\s*=\s*\{([^}]*)\}',body)
        ti=re.search(r'title\s*=\s*[{"](.*?)[}"]\s*,\s*\n',body,re.S)
        yield m.group(2), (d.group(1) if d else None), (re.sub(r'[{}\\]','',ti.group(1)) if ti else '')
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'bib-check/1.0'})
    for a in range(3):
        try: return urllib.request.urlopen(req,timeout=30).read().decode()
        except Exception: time.sleep(2+2*a)
    return None
n1=0
if not OFF:
    for src,t in (('bib_additions.bib',add),('corrected_entries.bib',cor)):
        for k,d,ti in entries(t):
            if not d: continue
            n1+=1
            if d.lower().startswith('10.48550/arxiv.'):
                aid=d[len('10.48550/arXiv.'):]
                x=get('http://export.arxiv.org/api/query?id_list='+urllib.parse.quote(aid)); tt=re.findall(r'<title>(.*?)</title>',x or '',re.S)
                got=tt[1] if len(tt)>1 else ''
            else:
                x=get('https://api.crossref.org/works/'+urllib.parse.quote(d,safe='/:;()'))
                m=json.loads(x)['message'] if x else {}; got=' '.join(m.get('title',[''])+m.get('subtitle',[]))
            a=set(w for w in norm(ti).split() if len(w)>3); b=set(w for w in norm(re.sub(r'<[^>]+>',' ',got)).split() if len(w)>3)
            ok=bool(got) and (not a or len(a&b)/max(1,len(a))>=0.6)
            if not ok: fail+=1
            out.append(f"DOI {'OK ' if ok else 'FAIL'} {src} {k} {d} | {re.sub(chr(10),' ',got)[:70]}")
            time.sleep(0.2)
blocks=json.load(open(os.path.join(B,'bib_insertions.json')))
for b in blocks:
    p=os.path.join(B,b['file']); L=rd(p).split('\n'); anc=b['anchor']
    c=L.count(anc); ok=c==1 and anc.count(b['old'])==1 and all(k in K for k in b['keys'])
    if not ok: fail+=1
    out.append(f"BLOCK {'OK ' if ok else 'FAIL'} {b['id']} {b['file']} anchor_count={c} keys={b['keys']}")
G=rd(os.path.join(B,'appendices','app_F_glossary.tex'))
book=''; main=rd(os.path.join(B,'main.tex'))
for f in re.findall(r'^\\input\{([^}]+)\}',main,re.M):
    p=os.path.join(B,f+'.tex')
    if os.path.exists(p) and 'app_F_glossary' not in f:
        t=rd(p); book+=t
        for g in re.findall(r'\\input\{([^}]+)\}',t):
            q=os.path.join(B,g+'.tex')
            if os.path.exists(q): book+=rd(q)
labels=set(re.findall(r'\\label\{([^}]+)\}',book+main))
ent=[l for l in G.split('\n') if l.startswith('\\textbf{')]
bad_ref=[r for r in set(re.findall(r'\\ref\{([^}]+)\}',G)) if r not in labels]
bad_cite=[k.strip() for m in re.finditer(r'\\cite\{([^}]*)\}',G) for k in m.group(1).split(',') if k.strip() not in K]
bal=[i for i,l in enumerate(ent) if re.sub(r'\\[{}$]','',l).count('{')!=re.sub(r'\\[{}$]','',l).count('}') or re.sub(r'\\\$','',l).count('$')%2]
nums=[n for l in ent for n in re.findall(r'(?<![A-Za-z_\\])\d+(?:\.\d+)?',re.sub(r'\\(ref|cite|label)\{[^}]*\}','',l)) if len(n)>1 and n not in book]
C=json.load(open(os.path.join(ROOT,'CANON','iam_canon.json')))
ret=[p for p,v in C['retired'].items() if re.search(p,G,0 if v['severity']=='block' else re.I)]
extra=r'the quantum-processor report|the semiconductor report|the methylation report|the cell-reading engine|Aristotel|MARGINAL|DETECTABLE|five substrates|Mechanism B|axis of evil|cusp-core|the author|[Cc]ohort|percentile|age-matched|\bAUC\b|IAMAtlas'
ret2=sorted(set(re.findall(extra,G)))
for name,v in (('unresolved refs',bad_ref),('missing cite keys',bad_cite),('unbalanced entries',bal),('untraced numbers',nums),('canon retired patterns',ret),('retired words',ret2)):
    if v: fail+=1
    out.append(f"GLOSSARY {'OK ' if not v else 'FAIL'} {name}: {v[:20]}")
out.append(f"glossary entries {len(ent)}; DOIs checked {n1}; blocks {len(blocks)}; FAIL {fail}")
open(os.path.join(ROOT,'docs','verification','scripts','verify_bib_glossary_output.txt'),'w').write('\n'.join(out)+'\n')
print('\n'.join(out[-8:])); sys.exit(1 if fail else 0)
