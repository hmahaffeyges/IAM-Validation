"""info_sphere_viewer.py - the information sphere as a picture (DEVELOPMENT, DEV-SPHERE-01; written 2026-10-23).

The address globes (globe_viewer.py) paint a map: every pixel of the sky is a stretch of the genome, colored by a value. The information sphere
is not a map of the genome. Each site is a point placed by how it behaves across IAM-Atlas v2.1's cell types (stage_info_sphere, construction
v5, the level taken out), so the picture is a field of light: points glow where many sites behave alike, and the empty sky stays dark, as the
masked sky does. Each point is colored by the cell its sites carry the identity of (stage_info_sphere.roles), the thing that put them together.
A faint wire sphere and its graticule turn with the points so the depth reads. Written by the one command:
  run_sample.py --build info-globe --build-out PAGE.html [--bundle B.json]
write(path, X, ...) builds a single self-contained HTML page (no outside files, no network)."""
import base64, json, numpy as np

CLEAR = 0.10   # a site carries one cell's identity when it departs from its own level in that cell by at least 0.10 more than in any other (stated)

TEX_H, TEX_W, SIGMA_DEG = 128, 256, 6.0   # the territory grid (equirectangular) and its kernel width (stated)

def territories(X, fam, nfam, H=TEX_H, W=TEX_W, sigma_deg=SIGMA_DEG, chunk=1024):
    """Each family's density of clearly marking sites in every direction of an equirectangular grid (a von Mises-Fisher kernel of width
    sigma_deg on the sphere). Returns (H, W, nfam). Direction of grid cell (lat, lon): (cos lat cos lon, sin lat, cos lat sin lon)."""
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi; lon = ((np.arange(W) + 0.5) / W) * 2 * np.pi - np.pi
    LA, LO = np.meshgrid(lat, lon, indexing="ij"); G = np.stack([np.cos(LA) * np.cos(LO), np.sin(LA), np.cos(LA) * np.sin(LO)], -1).reshape(-1, 3).astype("float32")
    k = 1.0 / np.radians(sigma_deg) ** 2; X = np.asarray(X, "float32"); O = np.zeros((nfam, len(X)), "float32"); O[np.asarray(fam), np.arange(len(X))] = 1.0
    out = np.zeros((len(G), nfam), "float32")
    for a in range(0, len(G), chunk): out[a:a + chunk] = np.exp(k * (G[a:a + chunk] @ X.T - 1.0)) @ O.T
    return out.reshape(H, W, nfam)

def territory_texture(T, counts, colors):
    """Color each direction by the family most over-represented there (its share of the local density over its share of all clearly
    marking sites: the enrichment), glowing with how strongly it is enriched and how dense the sites are; empty sky stays dark."""
    tot = T.sum(-1); N = np.maximum(np.asarray(counts, float), 1.0); E = (T / np.maximum(tot[..., None], 1e-9)) / (N / N.sum())
    Es = np.sort(E, -1); top = E.argmax(-1); emax = Es[..., -1]; q = np.quantile(tot, 0.97); dens = 1 - np.exp(-2.0 * tot / q)
    st = np.clip((emax - 1.2) / 3.0, 0, 1); lane = np.clip((emax / np.maximum(Es[..., -2], 1e-9) - 1.0) / 0.8, 0, 1) ** 0.8   # dark lanes where two families meet
    FC = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in colors])
    glow = 0.75 * dens * st ** 1.1 * (0.15 + 0.85 * lane); rgb = FC[top] * glow[..., None] + np.array([0.03, 0.035, 0.08]) * dens[..., None]
    return (np.clip(rgb, 0, 1) * 255).astype("u1"), top.astype("u1"), np.clip(np.round(emax * 20), 0, 255).astype("u1"), E

def territory_peaks(E, top, min_sep_deg=55.0, min_enrich=2.0, per_family=2):
    """Label places: for each family, up to per_family directions where it is most enriched (and leads), at least min_sep_deg apart."""
    H, W, F = E.shape; lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi; lon = ((np.arange(W) + 0.5) / W) * 2 * np.pi - np.pi
    LA, LO = np.meshgrid(lat, lon, indexing="ij"); G = np.stack([np.cos(LA) * np.cos(LO), np.sin(LA), np.cos(LA) * np.sin(LO)], -1)
    out = []; cs = np.cos(np.radians(min_sep_deg))
    for f in range(F):
        e = np.where(top == f, E[..., f], 0).ravel(); picked = []
        for idx in np.argsort(e)[::-1]:
            if e[idx] < min_enrich or len(picked) >= per_family: break
            g = G.reshape(-1, 3)[idx]
            if all(g @ p < cs for p in picked): picked.append(g); out.append([f, *np.round(g, 4).tolist(), round(float(e[idx]), 2)])
    return out

def _b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

def write(path, X, cell, sgn, size, margin, cells, cell_family, families, title, about, sample=None):
    """X: unit vectors (n x 3); cell, sgn, size, margin: stage_info_sphere.roles; cells: cell type names; cell_family: family index of each
    cell type; families: [(name, color, members)]. sample (optional): per-site departure of one reading from its healthy expectation, the
    sample layer (red: higher than expected, blue: lower; NaN where not measured)."""
    q = np.clip(np.round(np.asarray(X, float) * 32767), -32767, 32767).astype("<i2")
    st = np.clip(np.round(np.asarray(size, float) * 255 / 0.6), 0, 255).astype("u1")
    mg = np.clip(np.round(np.asarray(margin, float) * 255 / 0.4), 0, 255).astype("u1")
    D = dict(n=int(len(q)), xyz=_b64(q), cell=_b64(np.asarray(cell, "u1")), up=_b64((np.asarray(sgn) > 0).astype("u1")), st=_b64(st), mg=_b64(mg),
             cells=[str(c) for c in cells], cf=[int(f) for f in cell_family], fam=[[n, c] for n, c, _ in families],
             counts=np.bincount(np.asarray(cell_family)[np.asarray(cell)], minlength=len(families)).tolist())
    # cluster centers for the labels: each family's clearly marking sites, split by direction (a cell's lower and higher sites sit on opposite
    # sides once each site's level is taken out); a center is labeled only where its sites gather (R at least 0.5, 100 sites or more)
    Xa = np.asarray(X, float); fam_of = np.asarray(cell_family)[np.asarray(cell)]; clear = np.asarray(margin) >= CLEAR; ctr = []
    for k, (n, c, _) in enumerate(families):
        for up in (0, 1):
            m = clear & (fam_of == k) & ((np.asarray(sgn) > 0) == bool(up))
            if m.sum() >= 100:
                v = Xa[m].mean(0); R = float(np.linalg.norm(v))
                if R >= 0.5: ctr.append([k, up, *(v / R).round(4).tolist(), round(R, 3), int(m.sum())])
    T = territories(Xa[clear], fam_of[clear], len(families)); cc = np.bincount(fam_of[clear], minlength=len(families))
    tex, ttop, ten, E = territory_texture(T, cc, [c for _, c, _ in families])
    D["peaks"] = territory_peaks(E, ttop)
    D.update(tex=_b64(tex), ttop=_b64(ttop), ten=_b64(ten), th=TEX_H, tw=TEX_W, sigma=SIGMA_DEG)
    D["ctr"] = ctr; D["clear"] = CLEAR; D["ccounts"] = np.bincount(fam_of[clear], minlength=len(families)).tolist()
    if sample is not None:   # 2026-10-23 (plan 35.2): one sample against its own expected healthy self (stage_info_sphere.sample_reading)
        x = np.asarray(sample["x"], float); lit = np.asarray(sample["lit"], bool); sg = np.asarray(sample["sign"], int)
        D["smp"] = _b64(np.where(np.isfinite(x), np.clip(np.round(x * 127 / 4.0), -127, 127), -128).astype("i1"))
        D["lit"] = _b64(np.where(lit, np.where(sg > 0, 1, 2), 0).astype("u1")); D["smpfix"] = sample["fixed"]
        if lit.any():
            Tc = territories(Xa[lit], (sg[lit] < 0).astype(int), 2); Tc = Tc / max(float(Tc.max()), 1e-9)
            pos, neg = Tc[..., 0], Tc[..., 1]; ch = np.stack([pos + 0.15 * neg, 0.2 * (pos + neg), neg + 0.15 * pos], -1)
            D["chg"] = _b64((np.clip(ch, 0, 1) * 255).astype("u1"))
    html = PAGE.replace("__TITLE__", title).replace("__ABOUT__", about).replace("__DATA__", json.dumps(D))
    open(path, "w", encoding="utf-8").write(html); return dict(sites=int(len(q)), bytes=len(html))

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{margin:0;background:#03040a;color:#d8dcef;font:15px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:18px;display:flex;gap:22px;flex-wrap:wrap}
 .stage{flex:0 0 auto;position:relative} canvas{display:block;border-radius:50%;cursor:grab;touch-action:none}
 .side{flex:1 1 330px;min-width:300px} h1{font-size:20px;font-weight:600;margin:4px 0 2px;color:#fff} .sub{color:#8f96b8;font-size:13px;margin-bottom:12px}
 .dev{display:inline-block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#ffcf6e;border:1px solid #6b5a2e;border-radius:4px;padding:1px 6px;margin-left:6px}
 .legend{display:grid;grid-template-columns:1fr;gap:3px;margin:8px 0 12px} .lg{display:flex;align-items:center;gap:8px;padding:3px 6px;border-radius:5px;cursor:pointer;user-select:none}
 .lg:hover{background:#11152a} .lg.off{opacity:.28} .sw{width:11px;height:11px;border-radius:50%;box-shadow:0 0 8px currentColor;background:currentColor;flex:0 0 11px}
 .lg .n{margin-left:auto;color:#6f779c;font-size:12px} label{font-size:13px;color:#a9b0d0} select,input[type=range]{width:100%;margin:3px 0 10px;accent-color:#7aa2ff}
 select{background:#0c1022;color:#d8dcef;border:1px solid #262c4a;border-radius:5px;padding:4px} .row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}
 button{background:#0f1430;color:#cdd3f0;border:1px solid #2a3260;border-radius:5px;padding:4px 10px;cursor:pointer} button.on{background:#1f2c66;border-color:#4a5cc0}
 .about{color:#aab0cf;font-size:13.5px;border-top:1px solid #1b2040;margin-top:10px;padding-top:10px} .about p{margin:0 0 8px} .about h2{font-size:14.5px;color:#dfe3ff;margin:16px 0 6px} .about ol,.about ul{margin:0 0 8px;padding-left:20px} .about li{margin:0 0 5px} .about table{border-collapse:collapse;margin:4px 0 10px;font-size:12.5px} .about td,.about th{border:1px solid #232a52;padding:4px 6px;vertical-align:top;text-align:left} .about th{color:#dfe3ff} .read{font-size:12.5px;color:#8f96b8;min-height:1.4em}
</style></head><body><div class="wrap">
<div class="stage"><canvas id="c" width="720" height="720"></canvas></div>
<div class="side"><h1>__TITLE__<span class="dev">development</span></h1>
<div class="sub">Drag to turn. Each point of light is a site of the genome, placed by how it behaves across IAM-Atlas v2.1's 79 healthy cell types.</div>
<div class="row"><button id="all">Show every family</button><button id="far">Far side</button><button id="spin" class="on">Turning</button><button id="lab" class="on">Labels</button><button id="ter" class="on">Territories</button><button id="dst">Every site</button></div>
<div class="row"><button id="d2" class="on">Both sides</button><button id="d0">Unmethylated in their cell</button><button id="d1">Methylated in their cell</button></div>
<div class="legend" id="lg"></div>
<label>Light one cell type's sites</label><select id="one"><option value="-1">every cell type</option></select>
<label>Color a site when it stands apart in one cell by at least: <span id="mv"></span></label><input id="mgn" type="range" min="0" max="100" value="25">
<div id="lay" style="display:none"><label>Layer</label><select id="layer"><option value="smp">This sample against its own expected healthy self</option><option value="role">What groups the sites (cell identity)</option></select>
<label>Slide right to quiet everything but the change: <span id="fv"></span></label><input id="fd" type="range" min="0" max="100" value="0"><div class="read" id="fx"></div></div>
<div class="read" id="rd"></div>
<div class="about">__ABOUT__</div></div></div>
<script>
const D=__DATA__;
function dec(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer)}
const N=D.n,P=dec(D.xyz,Int16Array),CE=dec(D.cell,Uint8Array),ST=dec(D.st,Uint8Array),MG=dec(D.mg,Uint8Array),SM=D.smp?dec(D.smp,Int8Array):null;
const hex=h=>[parseInt(h.slice(1,3),16)/255,parseInt(h.slice(3,5),16)/255,parseInt(h.slice(5,7),16)/255];
const UPA=dec(D.up,Uint8Array),FC=D.fam.map(f=>hex(f[1])),CF=D.cf,on=D.fam.map(()=>true);let one=-1,far=false,spin=true,mth=D.clear,layer='role',dir=2,lab=true;
const lg=document.getElementById('lg');
D.fam.forEach((f,k)=>{const e=document.createElement('div');e.className='lg';e.innerHTML='<span class="sw" style="color:'+f[1]+'"></span>'+f[0]+'<span class="n">'+D.ccounts[k].toLocaleString()+'</span>';
 e.onclick=()=>{const solo=on.every((v,j)=>j===k?v:!v);if(solo){on.fill(true)}else{on.fill(false);on[k]=true}upd()};lg.appendChild(e)});
function upd(){[...lg.children].forEach((e,k)=>e.classList.toggle('off',!on[k]));}
const sel=document.getElementById('one');D.cells.map((c,i)=>[c,i]).sort().forEach(([c,i])=>{const o=document.createElement('option');o.value=i;o.textContent=c;sel.appendChild(o)});
sel.onchange=()=>{one=+sel.value};document.getElementById('all').onclick=()=>{on.fill(true);upd()};
const fb=document.getElementById('far');fb.onclick=()=>{far=!far;fb.classList.toggle('on',far)};const sb=document.getElementById('spin');sb.onclick=()=>{spin=!spin;sb.classList.toggle('on',spin)};
const mr=document.getElementById('mgn'),mv=document.getElementById('mv');const mset=()=>{mth=+mr.value/100*0.4;mv.textContent=mth.toFixed(2)+' more than in any other cell'+(mth?'; the rest drawn as dust':'')};mr.oninput=mset;mset();
const dB=[0,1,2].map(k=>document.getElementById('d'+k));dB.forEach((b,k)=>b.onclick=()=>{dir=k;dB.forEach((x,j)=>x.classList.toggle('on',j===k))});
const tb=document.getElementById('ter');tb.onclick=()=>{terr=!terr;tb.classList.toggle('on',terr)};const db=document.getElementById('dst');db.onclick=()=>{dust=!dust;db.classList.toggle('on',dust)};
const lb=document.getElementById('lab');lb.onclick=()=>{lab=!lab;lb.classList.toggle('on',lab)};
let fade=0;const LT=D.lit?dec(D.lit,Uint8Array):null,CHT=D.chg?dec(D.chg,Uint8Array):null;
if(SM){layer='smp';document.getElementById('lay').style.display='';document.getElementById('layer').onchange=e=>{layer=e.target.value};document.getElementById('fx').textContent=D.smpfix||'';
 const fd=document.getElementById('fd'),fv=document.getElementById('fv');const fs=()=>{fade=+fd.value/100;fv.textContent=fade===0?'the whole sphere':fade===1?'only the change':'the rest fading to dark'};fd.oninput=fs;fs()}
const TX=dec(D.tex,Uint8Array),TT=dec(D.ttop,Uint8Array),TE=dec(D.ten,Uint8Array),TH=D.th,TW=D.tw;let terr=true,dust=false,mx=-1,my=-1;
function texAt(x,y,z){const la=Math.asin(Math.max(-1,Math.min(1,y))),lo=Math.atan2(z,x);let v=(0.5-la/Math.PI)*TH-0.5,u=((lo+Math.PI)/(2*Math.PI))*TW-0.5;
 const v0=Math.max(0,Math.min(TH-1,Math.floor(v))),v1=Math.min(TH-1,v0+1),fv=Math.max(0,Math.min(1,v-v0));const u0=((Math.floor(u)%TW)+TW)%TW,u1=(u0+1)%TW,fu=u-Math.floor(u);
 const o=[0,0,0];for(const [vv,wv] of[[v0,1-fv],[v1,fv]])for(const [uu,wu] of[[u0,1-fu],[u1,fu]]){const k=(vv*TW+uu)*3,w=wv*wu;o[0]+=TX[k]*w;o[1]+=TX[k+1]*w;o[2]+=TX[k+2]*w}
 return[o,TT[v0*TW+u0],TE[v0*TW+u0]/20]}
const cv=document.getElementById('c'),cx=cv.getContext('2d'),W=cv.width,H=cv.height,R=W*0.45,C0=W/2;
const img=cx.createImageData(W,H),acc=new Float32Array(W*H*3);
let ry=0.6,rx=-0.35,drag=null;
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY,ry,rx];cv.setPointerCapture(e.pointerId);cv.style.cursor='grabbing'};
cv.onpointermove=e=>{if(!drag)return;ry=drag[2]+(e.clientX-drag[0])*0.008;rx=Math.max(-1.5,Math.min(1.5,drag[3]+(e.clientY-drag[1])*0.008))};
cv.onpointerup=()=>{drag=null;cv.style.cursor='grab'};cv.onmouseleave=()=>{mx=-1};cv.addEventListener('mousemove',e=>{const b=cv.getBoundingClientRect();mx=(e.clientX-b.left)*W/b.width;my=(e.clientY-b.top)*H/b.height});
function rot(){const cy=Math.cos(ry),sy=Math.sin(ry),cxx=Math.cos(rx),sx=Math.sin(rx);return[cy,0,sy, sx*sy,cxx,-sx*cy, -cxx*sy,sx,cxx*cy]}
function put(x,y,r,g,b,w){const i=x|0,j=y|0;if(i<1||j<1||i>=W-1||j>=H-1)return;let k=(j*W+i)*3;acc[k]+=r*w;acc[k+1]+=g*w;acc[k+2]+=b*w;const h=w*0.35;
 for(const d of[3,-3,W*3,-W*3]){acc[k+d]+=r*h;acc[k+d+1]+=g*h;acc[k+d+2]+=b*h}}
function frame(){if(spin&&!drag)ry+=0.0035;acc.fill(0);const M=rot();
 for(let a=0;a<6;a++){const ph=a*Math.PI/6;for(let t=0;t<200;t++){const th=t/200*2*Math.PI,x0=Math.cos(th)*Math.cos(ph),z0=Math.cos(th)*Math.sin(ph),y0=Math.sin(th);
  const X=M[0]*x0+M[1]*y0+M[2]*z0,Y=M[3]*x0+M[4]*y0+M[5]*z0,Z=M[6]*x0+M[7]*y0+M[8]*z0;put(C0+R*X,C0-R*Y,.35,.42,.75,Z>0?.10:.035)}}
 for(let a=1;a<6;a++){const la=-Math.PI/2+a*Math.PI/6,cl=Math.cos(la),sl=Math.sin(la);for(let t=0;t<220;t++){const th=t/220*2*Math.PI,x0=cl*Math.cos(th),z0=cl*Math.sin(th),y0=sl;
  const X=M[0]*x0+M[1]*y0+M[2]*z0,Y=M[3]*x0+M[4]*y0+M[5]*z0,Z=M[6]*x0+M[7]*y0+M[8]*z0;put(C0+R*X,C0-R*Y,.35,.42,.75,Z>0?.10:.035)}}
 const s=1/32767;let shown=0;
 for(let i=0;i<N;i++){const c=CE[i],f=CF[c];if(f<0)continue;const clear=MG[i]/255*0.4>=mth,match=on[f]&&(dir===2||UPA[i]===dir);
  const x0=P[3*i]*s,y0=P[3*i+1]*s,z0=P[3*i+2]*s,Z=M[6]*x0+M[7]*y0+M[8]*z0;if(Z<0&&!far)continue;
  const X=M[0]*x0+M[1]*y0+M[2]*z0,Y=M[3]*x0+M[4]*y0+M[5]*z0;let r,g,b,w;
  if(layer==='smp'&&SM){const v=SM[i],l=LT?LT[i]:0;if(l){r=l===1?1:.2;g=.25;b=l===2?1:.2;w=.6}else if(v===-128){if(!dust)continue;r=g=b=.3;w=.01*(1-fade)}else{const a=v/127;r=a>0?1:.4;g=.45;b=a<0?1:.4;w=(.015+.12*Math.abs(a))*(1-fade)}}
  else if(clear&&match){const col=FC[f];r=col[0];g=col[1];b=col[2];w=(terr?.09:.16)+(terr?.32:.6)*ST[i]/255;if(one>=0){if(c===one)w=.95;else w*=.06}}
  else{if(!dust)continue;r=.55;g=.6;b=.85;w=.018}
  if(Z<0)w*=.25;else w*=.55+.45*Z;put(C0+R*X,C0-R*Y,r,g,b,w);if(clear&&match)shown++}
 const d=img.data;let hov=null;for(let p=0,k=0;p<W*H;p++,k+=3){const xx=(p%W)-C0,yy=(p/W|0)-C0,rr=Math.sqrt(xx*xx+yy*yy)/R;
  let br=rr<1?0.018+0.03*(1-rr):Math.max(0,0.05*(1-(rr-1)*12));const q=p*4;
  if(terr&&rr<1){const u=xx/R,v=-yy/R,z=Math.sqrt(Math.max(0,1-u*u-v*v));const wx=M[0]*u+M[3]*v+M[6]*z,wy=M[1]*u+M[4]*v+M[7]*z,wz=M[2]*u+M[5]*v+M[8]*z;
   const [o,tt,te]=texAt(wx,wy,wz);const sh=0.55+0.45*z,bg=layer==='smp'?0.55*(1-fade):0.9;acc[k]+=o[0]/255*sh*bg;acc[k+1]+=o[1]/255*sh*bg;acc[k+2]+=o[2]/255*sh*bg;
   if(layer==='smp'&&CHT){const la=Math.asin(Math.max(-1,Math.min(1,wy))),lo=Math.atan2(wz,wx),vv=Math.max(0,Math.min(TH-1,((0.5-la/Math.PI)*TH)|0)),uu=Math.max(0,Math.min(TW-1,(((lo+Math.PI)/(2*Math.PI))*TW)|0)),q3=(vv*TW+uu)*3;
    acc[k]+=CHT[q3]/255*sh*1.3;acc[k+1]+=CHT[q3+1]/255*sh*1.3;acc[k+2]+=CHT[q3+2]/255*sh*1.3}
   if(mx>=0&&Math.abs((p%W)-mx)<0.5&&Math.abs((p/W|0)-my)<0.5)hov=[tt,te]}
  d[q]=255*(1-Math.exp(-(acc[k]*1.6+br*0.5)));d[q+1]=255*(1-Math.exp(-(acc[k+1]*1.6+br*0.6)));d[q+2]=255*(1-Math.exp(-(acc[k+2]*1.6+br*1.4)));d[q+3]=255}
 cx.putImageData(img,0,0);
 if(lab){cx.font='12px -apple-system,Helvetica,Arial,sans-serif';cx.textAlign='center';for(const t of D.peaks){const [k,x0,y0,z0]=t;const up=-1;if(!on[k])continue;
  const X=M[0]*x0+M[1]*y0+M[2]*z0,Y=M[3]*x0+M[4]*y0+M[5]*z0,Z=M[6]*x0+M[7]*y0+M[8]*z0;if(Z<0.15)continue;const px=C0+R*X*0.97,py=C0-R*Y*0.97;
  cx.globalAlpha=Math.min(1,(Z-0.15)*2.2);cx.fillStyle='rgba(3,4,10,.55)';const txt=D.fam[k][0];const tw=cx.measureText(txt).width;cx.fillRect(px-tw/2-4,py-9,tw+8,16);
  cx.fillStyle=D.fam[k][1];cx.fillText(txt,px,py+3)}cx.globalAlpha=1}
 document.getElementById('rd').textContent=(hov&&hov[1]>1.2?'Pointer: '+D.fam[hov[0]][0]+' territory, '+hov[1].toFixed(1)+' times its share. ':'')+shown.toLocaleString()+' sites colored in view'+(far?' (both sides)':' (the near side)')+'; \u25BC unmethylated in their cell, \u25B2 methylated';requestAnimationFrame(frame)}
upd();frame();
</script></body></html>"""
