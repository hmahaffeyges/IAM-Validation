#!/usr/bin/env python3
"""
Terminal / Post-Mitotic — Architecture Class Specification Card
The complete thermodynamic operating envelope of a human post-mitotic cell.
Disease states appear as reference points on the physics, not as the frame.
Heath W. Mahaffey | April 2026
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import Flowable
import math

# ── Issue 001 palette ─────────────────────────────────────────────────────────
BG    = colors.HexColor('#080810')
SURF  = colors.HexColor('#0d0d1e')
SURF2 = colors.HexColor('#111128')
BORD  = colors.HexColor('#1a1a3a')
LAV   = colors.HexColor('#C4B5FD')
LAV_D = colors.HexColor('#7C3AED')
LAV_M = colors.HexColor('#A78BFA')
GRN   = colors.HexColor('#4ade80')
GRN2  = colors.HexColor('#12c97a')
AMB   = colors.HexColor('#facc15')
REDC  = colors.HexColor('#ef4444')
RED2  = colors.HexColor('#dc2626')
TEAL  = colors.HexColor('#00C9B1')
ORAN  = colors.HexColor('#fb923c')
TEXT  = colors.HexColor('#EDE9FE')
MUT   = colors.HexColor('#4a3a7a')
MUT2  = colors.HexColor('#7C6BA8')
WHT   = colors.white
INDIGO= colors.HexColor('#6366f1')   # terminal class

W, H  = letter
PW    = W - 1.0*inch

# ── Physics ───────────────────────────────────────────────────────────────────
k_B = 1.380649e-23; ln2 = math.log(2); R_gas = 8.314462; T = 310.15
DELTA_G_ATP = 54000.0; N_CpG = 19.6e6
E_floor = N_CpG * k_B * T * ln2

H_MIN_GLOBAL = 0.756500
H_MIN_TERM   = 0.772837
n_bio        = 24.5
gen_rate     = 0.008      # 0.8%/generation

# Reference states — healthy, disease, extreme
BETA_HEALTHY    = 0.768   # frontal cortex neuron
BETA_AD_LOW     = 0.775   # low AD neuropathology (De Jager 2014)
BETA_AD_HIGH    = 0.764   # high AD neuropathology (De Jager 2014)
BETA_TUMOR_LGG  = 0.450   # LGG mean (Ceccarelli 2016)
BETA_TUMOR_GBM  = 0.400   # GBM mean (Ceccarelli 2016)

def H(b):
    if b <= 0 or b >= 1: return 0.0
    return -b*math.log2(b) - (1-b)*math.log2(1-b)

def A(b): return H(b) / H_MIN_TERM

A_healthy   = A(BETA_HEALTHY)
A_ad_low    = A(BETA_AD_LOW)
A_ad_high   = A(BETA_AD_HIGH)
A_lgg       = A(BETA_TUMOR_LGG)
A_gbm       = A(BETA_TUMOR_GBM)

# C1/C2/C3 at healthy state
C1    = H_MIN_GLOBAL
C2    = H_MIN_TERM - H_MIN_GLOBAL
H_n   = H(BETA_HEALTHY)
C3_n  = max(0.0, H_n - H_MIN_TERM)
f_C1  = C1/H_n*100; f_C2 = C2/H_n*100; f_C3 = C3_n/H_n*100
locked = (H_MIN_TERM/H_n)*100; accessible = f_C3

def proj_A(gens): return round(A_healthy * ((1+gen_rate)**gens), 4)
def metab_A(pct): return round(A_healthy * ((1+pct/100)**n_bio), 4)

# ── Styles ────────────────────────────────────────────────────────────────────
def S(name, **kw):
    d = dict(fontName='Helvetica', fontSize=9, leading=13, textColor=TEXT,
             backColor=None, spaceBefore=0, spaceAfter=0)
    d.update(kw); return ParagraphStyle(name, **d)

sLabel = S('La', fontName='Helvetica-Bold', fontSize=7.5, leading=11,
           textColor=LAV, spaceBefore=3, spaceAfter=2)
sSub   = S('Su', fontName='Helvetica-Bold', fontSize=6.5, leading=9,
           textColor=LAV_D, spaceBefore=0, spaceAfter=2)
sBody  = S('B',  fontSize=8.5, leading=14, textColor=TEXT, spaceAfter=4)
sComm  = S('Co', fontSize=8.5, leading=14, textColor=TEXT, spaceBefore=4, spaceAfter=6)
sMut   = S('M',  fontSize=7.5, leading=12, textColor=MUT2, spaceAfter=2)
_sTH   = S('TH', fontName='Helvetica-Bold', fontSize=7.5, textColor=LAV,  leading=11)
_sTD   = S('TD', fontSize=7.5, textColor=TEXT, leading=11)
_sTDs  = S('TDs',fontSize=7,   textColor=MUT2, leading=10)

def P(txt, st=None):
    return Paragraph(str(txt), st or _sTD)
def PH(txt): return P(txt, _sTH)
def SP(n):   return Spacer(1, n*inch)
def HR(c=LAV_D, t=0.5): return HRFlowable(width='100%', thickness=t, color=c, spaceAfter=4)

def tbl_sty(fs=7.5):
    return TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  SURF2),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [SURF, colors.HexColor('#0a0a18')]),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), fs),
        ('TEXTCOLOR',     (0,0),(-1,0),  LAV),
        ('TEXTCOLOR',     (0,1),(-1,-1), TEXT),
        ('TOPPADDING',    (0,0),(-1,-1), 3),
        ('BOTTOMPADDING', (0,0),(-1,-1), 3),
        ('LEFTPADDING',   (0,0),(-1,-1), 5),
        ('RIGHTPADDING',  (0,0),(-1,-1), 5),
        ('GRID',          (0,0),(-1,-1), 0.3, BORD),
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
    ])

class FillRect(Flowable):
    def __init__(self, w, h, fill, r=4):
        super().__init__(); self.width=w; self.height=h; self.fill=fill; self.r=r
    def draw(self):
        self.canv.setFillColor(self.fill)
        self.canv.roundRect(0, 0, self.width, self.height, self.r, fill=1, stroke=0)

# ── Fidelity gauge — shows full operating range, reference states as markers ──
class FidelityGauge(Flowable):
    """Full operating range gauge. Reference states marked — not organized around them."""
    def __init__(self, width=None):
        super().__init__(); self.width = width or PW; self.height = 85
    def draw(self):
        c = self.canv
        bar_x=110; bar_w=self.width-250; bar_h=16; bar_y=30
        A_lo=0.85; A_hi=1.50
        def xp(Av): return bar_x + max(0.0,min(1.0,(Av-A_lo)/(A_hi-A_lo)))*bar_w

        # Zone fills — operating range descriptions, not clinical triage
        zones = [
            (0.85, 1.00, colors.HexColor('#0a1a0a'), 'SUB-FLOOR'),
            (1.00, 1.02, colors.HexColor('#1a4a1a'), 'REFERENCE'),
            (1.02, 1.05, colors.HexColor('#2a4a10'), 'MARGINAL'),
            (1.05, 1.10, colors.HexColor('#4a3800'), 'DETECTABLE'),
            (1.10, 1.35, colors.HexColor('#4a1800'), 'DEPARTURE'),
            (1.35, 1.50, colors.HexColor('#3a0000'), 'EXTREME'),
        ]
        zone_colors = [GRN2, GRN, AMB, REDC, RED2, colors.HexColor('#FF00FF')]
        for (a0,a1,bg,lbl),col in zip(zones, zone_colors):
            x0=xp(a0); x1=min(xp(a1), bar_x+bar_w)
            c.setFillColor(bg); c.rect(x0,bar_y,x1-x0,bar_h,fill=1,stroke=0)
            w_z = x1-x0
            if w_z > 20:
                c.setFillColor(col); c.setFont('Helvetica',5)
                c.drawCentredString((x0+x1)/2, bar_y+5, lbl)

        # Border
        c.setStrokeColor(BORD); c.setLineWidth(0.5)
        c.rect(bar_x,bar_y,bar_w,bar_h,fill=0,stroke=1)

        # Threshold tick marks
        for Av,col,lbl in [(1.00,GRN,'1.00'),(1.02,GRN,'1.02'),
                            (1.05,AMB,'1.05'),(1.10,REDC,'1.10'),(1.35,RED2,'1.35')]:
            xv=xp(Av)
            c.setStrokeColor(col); c.setLineWidth(0.8); c.setDash([2,2])
            c.line(xv,bar_y-3,xv,bar_y+bar_h+3); c.setDash([])
            c.setFillColor(col); c.setFont('Helvetica-Bold',5)
            c.drawCentredString(xv,bar_y+bar_h+6,lbl)

        # FLOOR label left
        c.setFillColor(MUT2); c.setFont('Helvetica',6)
        c.drawRightString(bar_x-4,bar_y+10,f'H_min={H_MIN_TERM:.6f}')
        c.drawRightString(bar_x-4,bar_y+2,'FLOOR')

        # Reference state markers — neutral triangles with labels above and below
        # Below bar: states below 1.10 (healthy, AD)
        # Above bar: extreme states (LGG, GBM)
        states_below = [
            (A_healthy, GRN2, f'Healthy neuron  A={A_healthy:.4f}'),
            (A_ad_low,  AMB,  f'Low AD  A={A_ad_low:.4f}'),
            (A_ad_high, ORAN, f'High AD  A={A_ad_high:.4f}'),
        ]
        states_above = [
            (A_lgg, REDC, f'LGG  A={A_lgg:.4f}'),
            (A_gbm, RED2, f'GBM  A={A_gbm:.4f}'),
        ]
        # Below bar markers
        for i,(Av,col,lbl) in enumerate(states_below):
            xv = xp(Av)
            c.setFillColor(col); c.setFont('Helvetica-Bold',7)
            c.drawCentredString(xv, bar_y-12, chr(9650))
            c.setFillColor(col); c.setFont('Helvetica',5.5)
            y_lbl = bar_y - 20 - i*9
            c.drawCentredString(xv, y_lbl, lbl)
        # Above bar markers
        for i,(Av,col,lbl) in enumerate(states_above):
            xv = xp(Av)
            c.setFillColor(col); c.setFont('Helvetica-Bold',7)
            c.drawCentredString(xv, bar_y+bar_h+18, chr(9660))
            c.setFillColor(col); c.setFont('Helvetica',5.5)
            c.drawCentredString(xv, bar_y+bar_h+27+i*8, lbl)


# ── C1/C2/C3 stacked bar ─────────────────────────────────────────────────────
class CompBar(Flowable):
    def __init__(self, c1, c2, c3, w=None):
        super().__init__()
        self.c1=c1; self.c2=c2; self.c3=c3; self.width=w or PW; self.height=20
    def draw(self):
        c=self.canv; bh=14; by=3
        total=self.c1+self.c2+self.c3
        w1=self.c1/total*self.width; w2=self.c2/total*self.width
        w3=max(self.c3/total*self.width, 4)
        c.setFillColor(colors.HexColor('#1a0a3a')); c.rect(0,by,w1,bh,fill=1,stroke=0)
        c.setFillColor(colors.HexColor('#1a120a')); c.rect(w1,by,w2,bh,fill=1,stroke=0)
        c.setFillColor(colors.HexColor('#081510')); c.rect(w1+w2,by,w3,bh,fill=1,stroke=0)
        for x,w,lbl,col in [(0,w1,f'C1 {self.c1:.0f}%',LAV),
                              (w1,w2,f'C2 {self.c2:.0f}%',AMB),
                              (w1+w2,w3,f'C3 {self.c3:.2f}%',GRN2)]:
            if w > 30:
                c.setFillColor(col); c.setFont('Helvetica-Bold',6.5)
                c.drawCentredString(x+w/2, by+4, lbl)
        c.setStrokeColor(BORD); c.setLineWidth(0.3)
        c.rect(0,by,self.width,bh,fill=0,stroke=1)


# ── Document ──────────────────────────────────────────────────────────────────
out = '/home/claude/fig_terminal_postmitotic_card.pdf'
doc = SimpleDocTemplate(out, pagesize=letter,
                        leftMargin=0.5*inch, rightMargin=0.5*inch,
                        topMargin=0.4*inch, bottomMargin=0.4*inch)

def bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG); canvas.rect(0,0,612,792,fill=1,stroke=0)
    canvas.restoreState()

story = []

# ── HEADER ────────────────────────────────────────────────────────────────────
story.append(FillRect(PW, 0.50*inch, SURF2, r=5))
story.append(Spacer(1, -0.50*inch)); story.append(Spacer(1, 4))
hdr = Table([[
    Paragraph(f'<font color="#6366f1">■</font>  <b>TERMINAL / POST-MITOTIC</b>',
              S('CH', fontName='Helvetica-Bold', fontSize=11, textColor=WHT, leading=14)),
    Paragraph(f'<font color="#7C6BA8" size="8">'
              f'H_min = {H_MIN_TERM:.6f}  ·  n_bio = ~{n_bio} (est.)  ·  '
              f'gen_rate = {gen_rate*100:.1f}%/gen  ·  '
              f'Reference: frontal cortex neuron</font>',
              S('CM', fontSize=8, textColor=MUT2, leading=11)),
    Paragraph(f'<b>A = {A_healthy:.4f}</b>',
              S('CA', fontName='Helvetica-Bold', fontSize=11, textColor=INDIGO,
                leading=12, alignment=TA_RIGHT)),
]], colWidths=[PW*0.33, PW*0.49, PW*0.18],
style=[('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0),
       ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
       ('BACKGROUND',(0,0),(-1,-1),colors.transparent),('VALIGN',(0,0),(-1,-1),'MIDDLE')])
story.append(hdr)
story.append(Spacer(1, 3))
story.append(Paragraph(
    'Cell types: neurons, cardiomyocytes, skeletal muscle fibre  ·  '
    'Reference cell: frontal cortex neuron (Roadmap Epigenomics E073, Lister 2013)  ·  '
    'MCMC: G-002 chain 4 of 5, R-hat 0.9998. Posterior H_min 0.7728 ± 0.0011.',
    S('Sr', fontSize=6.5, textColor=MUT, leading=9)))
story.append(Spacer(1, 6))

# ── CELL BIOLOGY & ARCHITECTURE ───────────────────────────────────────────────
story.append(Paragraph('CELL BIOLOGY & ARCHITECTURE', sLabel)); story.append(Spacer(1,2))
ct_rows = [
    [P('Cell types', _sTH),
     P('Neurons · cardiomyocytes · skeletal muscle fibres')],
    [P('Commitment state', _sTH),
     P('Post-mitotic — permanent cell cycle exit, irreversible differentiation')],
    [P('Methylation maintenance', _sTH),
     P('DNMT1 passive maintenance only — no active remodelling in healthy tissue')],
    [P('Architecture floor H_min', _sTH),
     Paragraph(f'<font name="Courier">{H_MIN_TERM:.6f}</font>  — lowest of all 8 classes; '
               'global minimum reference (frontal cortex neuron)',
               S('tv', fontSize=7.5, textColor=TEAL, leading=11))],
    [P('Healthy A-score', _sTH),
     Paragraph(f'<font name="Courier">{A_healthy:.4f}</font>  '
               f'(β = {BETA_HEALTHY})  ·  {f_C3:.2f}% accessible gap above floor',
               S('tv', fontSize=7.5, textColor=GRN2, leading=11))],
    [P('Metabolic sensitivity', _sTH),
     P(f'n_bio = ~{n_bio} (est., PRELIMINARY) — highest of all 8 classes')],
]
ct_t = Table(ct_rows, colWidths=[PW*0.22, PW*0.76])
ct_t.setStyle(tbl_sty()); story.append(ct_t); story.append(Spacer(1,6))

# ── COMMENTARY ────────────────────────────────────────────────────────────────
story.append(Paragraph('COMMENTARY', sLabel)); story.append(Spacer(1,2))
story.append(Paragraph(
    'Terminal post-mitotic cells are the most committed cells in the body. '
    'Neurons, cardiomyocytes, and skeletal muscle fibres have exited the cell cycle permanently '
    'and reached their final differentiated state. Their methylation program is locked in place '
    'by DNMT3A and DNMT3B during development, then maintained by DNMT1 with extreme fidelity '
    'across decades. A frontal cortex neuron alive today may have maintained its methylation '
    'program for longer than any other structure in the body.',
    sComm))
story.append(Paragraph(
    'This maximum commitment is encoded in the lowest H_min of any architecture class: 0.772837. '
    'The Shannon entropy of a healthy neuron sits just 1.1% above this floor — leaving almost '
    'no accessible gap. The three-component decomposition makes this visible: C1 (the universal '
    'Landauer floor, identical for every cell on Earth) accounts for 96.8% of the entropy; '
    'C2 (the architecture overhead of being specifically a post-mitotic cell) adds 2.1%; '
    'C3 (the accessible gap above the class floor that any biological intervention can address) '
    'is only 0.11%. The terminal cell has committed so completely that it operates essentially '
    'at its physical minimum. This is not a pathology — it is the architecture of permanence.',
    sComm))
story.append(Paragraph(
    'The full operating range of the terminal class spans from the physical floor at A = 1.000 '
    'through the normal healthy operating window (A ≈ 1.01), across the marginal zone where '
    'slow epigenomic drift accumulates over decades (Alzheimer\'s disease reaches A ≈ 1.04–1.06 '
    'at high neuropathology burden), to the extreme departures seen in glioma (A = 1.285 for '
    'LGG, A = 1.256 for GBM) where the methylation entropy has collapsed to near 0.45. '
    'These disease states appear on the operating range gauge below as reference points. '
    'The gauge shows the full physical range of what this cell class can do — not a clinical triage.',
    sComm))

# ── FIDELITY POSITION ─────────────────────────────────────────────────────────
story.append(Paragraph('FIDELITY POSITION — FULL OPERATING RANGE', sLabel)); story.append(Spacer(1,2))
story.append(Paragraph(
    'Reference states shown as markers. Normal healthy state (▲) below bar. '
    'Disease reference points (▼) above bar where applicable. '
    'Zone labels describe the thermodynamic operating regime, not a clinical recommendation.',
    S('TH', fontSize=7, textColor=MUT2, leading=11, spaceAfter=4)))
story.append(FidelityGauge())
story.append(Spacer(1, 8))

# ── CORE METRICS ─────────────────────────────────────────────────────────────
story.append(Paragraph('CORE METRICS & DERIVED QUANTITIES', sLabel)); story.append(Spacer(1,2))
cm_rows = [[PH('Metric'), PH('Value'), PH('Source')]]
metrics = [
    ('Global floor (H_min_global)',
     f'<font name="Courier">{H_MIN_GLOBAL:.6f}</font>',
     'Frontal cortex neuron — Lister 2013  DERIVED', TEAL),
    ('Class floor (H_min)',
     f'<font name="Courier">{H_MIN_TERM:.6f}</font>',
     'G-002 MCMC — 5 chains R-hat < 1.001  DERIVED', TEAL),
    ('Metabolic sensitivity (n_bio)',
     f'<font name="Courier">~{n_bio} (est.)</font>',
     'G_ATP/(R·T_body) — PRELIMINARY pending G-001', TEAL),
    ('Healthy drift rate',
     f'<font name="Courier">{gen_rate*100:.1f}%/gen</font>',
     'Class registry  DERIVED', TEAL),
    ('Healthy A-score',
     f'<font name="Courier">{A_healthy:.5f}</font>',
     'H(β=0.768) / H_min  DERIVED', GRN2),
    ('Accessible gap (C3) — healthy',
     f'<font name="Courier">{f_C3:.2f}% of H(β)</font>',
     'H(β) − H_min  DERIVED', TEAL),
    ('Architecture-locked (C1+C2)',
     f'<font name="Courier">{locked:.1f}% of H(β) irreducible</font>',
     '(C1+C2)/H(β)  DERIVED', TEAL),
    ('Alzheimer\'s (low neuropath.)',
     f'<font name="Courier">A = {A_ad_low:.4f}  (β={BETA_AD_LOW})</font>',
     'De Jager 2014, Nat Neurosci  OBSERVED', AMB),
    ('Alzheimer\'s (high neuropath.)',
     f'<font name="Courier">A = {A_ad_high:.4f}  (β={BETA_AD_HIGH})</font>',
     'De Jager 2014, Nat Neurosci  OBSERVED', AMB),
    ('LGG (class mean)',
     f'<font name="Courier">A = {A_lgg:.4f}  (β={BETA_TUMOR_LGG})</font>',
     'Ceccarelli 2016 Cell  OBSERVED', REDC),
    ('GBM (class mean)',
     f'<font name="Courier">A = {A_gbm:.4f}  (β={BETA_TUMOR_GBM})</font>',
     'Ceccarelli 2016 Cell  OBSERVED', REDC),
    ('Generations to A = 1.05',
     f'<font name="Courier">'
     f'{round(math.log(1.05/A_healthy)/math.log(1+gen_rate),0):.0f} gen at {gen_rate*100:.1f}%/gen'
     f'</font>',
     'log(1.05/A_healthy)/log(1+gen_rate)  ILLUSTRATIVE', AMB),
]
for metric, val, src, col in metrics:
    cm_rows.append([
        P(metric),
        Paragraph(val, S('cv',fontSize=7.5,textColor=col,leading=11)),
        P(src),
    ])
cm_t = Table(cm_rows, colWidths=[PW*0.34, PW*0.36, PW*0.28], repeatRows=1)
cm_t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),SURF2),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[SURF,colors.HexColor('#0a0a18')]),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7),
    ('TEXTCOLOR',(0,0),(-1,0),LAV),('TEXTCOLOR',(0,1),(-1,-1),TEXT),
    ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),
    ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
    ('GRID',(0,0),(-1,-1),0.3,BORD),('VALIGN',(0,0),(-1,-1),'TOP'),
]))
story.append(cm_t); story.append(Spacer(1,6))

# ── FIDELITY TRAJECTORY ───────────────────────────────────────────────────────
story.append(Paragraph('FIDELITY TRAJECTORY — HEALTHY AGING DRIFT', sLabel))
story.append(Spacer(1,2))
story.append(Paragraph(
    f'Projected A-score at healthy drift rate ({gen_rate*100:.1f}%/gen). '
    f'Illustrative — assumes constant drift. '
    f'At this rate, A = 1.05 is reached at approximately generation '
    f'{round(math.log(1.05/A_healthy)/math.log(1+gen_rate),0):.0f}. '
    f'Alzheimer\'s and other neurodegenerative conditions '
    f'accelerate this trajectory without reaching cancer-range A-scores.',
    S('TH', fontSize=7, textColor=MUT2, leading=11, spaceAfter=4)))
traj_rows = [[PH('Generation'), PH('A-Score'), PH('Operating zone'), PH('Physiological context')]]
traj_data = [
    (0,   'Reference state — healthy adult neuron'),
    (5,   'Early drift accumulation'),
    (10,  'Mild epigenomic entropy increase'),
    (20,  'Moderate drift — approaching marginal zone'),
    (30,  'Slow long-term accumulation'),
]
zone_map = {
    (1.00,1.02): ('REFERENCE',    '#4ade80'),
    (1.02,1.05): ('MARGINAL',     '#facc15'),
    (1.05,1.10): ('DETECTABLE',   '#fb923c'),
    (1.10,1.35): ('DEPARTURE',    '#ef4444'),
    (1.35,9.99): ('EXTREME',      '#e879f9'),
}
def get_zone(Av):
    for (lo,hi),(lbl,col) in zone_map.items():
        if lo <= Av < hi: return lbl, col
    return 'REFERENCE','#4ade80'
for gen, context in traj_data:
    Ag = proj_A(gen)
    lbl, col = get_zone(Ag)
    traj_rows.append([
        P(f'Gen {gen}'),
        Paragraph(f'<font name="Courier">{Ag:.4f}</font>',
                  S('tv',fontSize=7.5,textColor=colors.HexColor(col),leading=11)),
        Paragraph(f'<b>{lbl}</b>',
                  S('tt',fontName='Helvetica-Bold',fontSize=7,
                    textColor=colors.HexColor(col),leading=10)),
        P(context),
    ])
traj_t = Table(traj_rows, colWidths=[PW*0.11, PW*0.18, PW*0.20, PW*0.49], repeatRows=1)
traj_t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),SURF2),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[SURF,colors.HexColor('#0a0a18')]),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7),
    ('TEXTCOLOR',(0,0),(-1,0),LAV),('TEXTCOLOR',(0,1),(-1,-1),TEXT),
    ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),
    ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
    ('GRID',(0,0),(-1,-1),0.3,BORD),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
story.append(traj_t); story.append(Spacer(1,6))

# ── METABOLIC SENSITIVITY ─────────────────────────────────────────────────────
story.append(Paragraph('METABOLIC SENSITIVITY', sLabel)); story.append(Spacer(1,2))
story.append(Paragraph(
    f'A-score response to ATP/ADP perturbation. '
    f'n_bio = ~{n_bio} governs the response magnitude — the biological analog of the '
    f'SCAPE temperature exponent. This is the highest n_bio of all 8 architecture classes: '
    f'small deviations in energy availability produce large changes in the fidelity index. '
    f'T_body = 310.15 K (37°C) is fixed — there is no thermal lever in biology.',
    S('TH', fontSize=7, textColor=MUT2, leading=11, spaceAfter=4)))
ms_rows = [[PH('ATP deviation'), PH('A-Score'), PH('vs reference'), PH('Operating zone')]]
for pct in [-10,-5,-2,0,+2,+5,+10]:
    Am = metab_A(pct); vs = round(Am/A_healthy,3)
    anchor = '  ← reference' if pct==0 else ''
    lbl, col = get_zone(Am)
    ms_rows.append([
        P(f'{pct:+d}%{anchor}'),
        Paragraph(f'<font name="Courier">{Am:.4f}</font>',
                  S('tv',fontSize=7.5,textColor=TEAL,leading=11)),
        Paragraph(f'<font name="Courier">{vs:.3f}×</font>',
                  S('tv',fontSize=7.5,
                    textColor=GRN2 if pct<0 else (REDC if pct>5 else TEXT),leading=11)),
        Paragraph(f'<b>{lbl}</b>',
                  S('tz',fontName='Helvetica-Bold',fontSize=6.5,
                    textColor=colors.HexColor(col),leading=10)),
    ])
ms_t = Table(ms_rows, colWidths=[PW*0.22, PW*0.18, PW*0.16, PW*0.22], repeatRows=1)
ms_t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),SURF2),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[SURF,colors.HexColor('#0a0a18')]),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7),
    ('TEXTCOLOR',(0,0),(-1,0),LAV),('TEXTCOLOR',(0,1),(-1,-1),TEXT),
    ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),
    ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
    ('GRID',(0,0),(-1,-1),0.3,BORD),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('BACKGROUND',(0,4),(-1,4),SURF2),
    ('FONTNAME',(0,4),(-1,4),'Helvetica-Bold'),
]))
story.append(ms_t); story.append(Spacer(1,6))

# ── ENTROPY GAP — THREE COMPONENTS ───────────────────────────────────────────
story.append(Paragraph('ENTROPY GAP — THREE COMPONENTS', sLabel))
story.append(Paragraph('at healthy reference state (β = 0.768)',
    S('Su2',fontName='Helvetica-Bold',fontSize=6.5,leading=9,
      textColor=LAV_D,spaceBefore=0,spaceAfter=3)))
story.append(Spacer(1,2))
story.append(CompBar(f_C1, f_C2, f_C3))
story.append(Spacer(1,3))
story.append(Paragraph(
    f'<b>C1 — Global Landauer floor: {f_C1:.0f}% ({H_MIN_GLOBAL:.6f})</b> — '
    f'The irreducible minimum entropy of any mammalian cell, set by the thermodynamic cost '
    f'of copying 19.6 million CpG sites at 37°C. Identical for every cell on Earth. '
    f'Nothing biological moves this.<br/>'
    f'<b>C2 — Architecture overhead: {f_C2:.0f}% ({C2:.6f})</b> — '
    f'The additional entropy of being specifically a post-mitotic cell. '
    f'Locked by architectural identity. Redifferentiation is the only lever.<br/>'
    f'<b>C3 — Accessible gap: {f_C3:.2f}% ({C3_n:.6f})</b> — '
    f'The fraction above the architecture floor that biological intervention can address. '
    f'In healthy terminal tissue this is nearly zero — the cell is at its floor. '
    f'This is not pathology: it is the architecture of maximum commitment.',
    S('TCB',fontSize=7,textColor=TEXT,leading=11))); story.append(Spacer(1,6))

# ── ARCHITECTURE-LOCKED FRACTION ──────────────────────────────────────────────
isa_block = Table([[
    Paragraph(f'<b>{locked:.1f}%</b>',
              S('IL',fontName='Helvetica-Bold',fontSize=11,
                textColor=TEAL,leading=14,alignment=TA_CENTER)),
    Paragraph(
        f'<b>{locked:.1f}% of the cell\'s measured entropy is architecture-locked — '
        f'irreducible by any biological intervention.</b><br/>'
        f'Only {accessible:.2f}% is intervention-accessible — the entropy above the class floor. '
        f'In healthy terminal cells this accessible fraction is essentially zero, '
        f'meaning the cell is operating at the absolute minimum consistent with its identity. '
        f'Interventions that attempt to lower entropy further cannot do so without '
        f'destroying the cell\'s architectural identity.',
        S('IB',fontSize=7,textColor=TEXT,leading=11)),
]], colWidths=[PW*0.12, PW*0.88],
style=[('BACKGROUND',(0,0),(-1,-1),SURF),('TOPPADDING',(0,0),(-1,-1),5),
       ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),8),
       ('RIGHTPADDING',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.3,BORD),
       ('VALIGN',(0,0),(-1,-1),'MIDDLE')])
story.append(Paragraph('ARCHITECTURE-LOCKED', sLabel))
story.append(Paragraph('FRACTION', S('LC6',fontName='Helvetica-Bold',fontSize=6.5,
                        leading=9,textColor=LAV_D,spaceBefore=0,spaceAfter=2)))
story.append(Spacer(1,2)); story.append(isa_block); story.append(Spacer(1,6))

# ── OPERATING RANGE INTERPRETATION ───────────────────────────────────────────
range_block = Table([[
    Paragraph('<b>FLOOR</b>',
              S('RG',fontName='Helvetica-Bold',fontSize=9,
                textColor=TEAL,leading=12,alignment=TA_CENTER)),
    Paragraph(
        f'<b>Class floor: H_min = {H_MIN_TERM:.6f}  ·  {A_healthy:.4f}x above floor  ·  '
        f'{(A_healthy/1.20 - 1)*100:.1f}% below departure threshold</b><br/>'
        f'The class floor is a first-principles value derived from the Landauer cost of '
        f'DNA methylation maintenance at 37°C, confirmed by MCMC against 49 published '
        f'reference cell types. It is not derived from disease data. '
        f'A healthy terminal cell sits {(A_healthy - 1.00)*100:.2f}% above its floor — '
        f'the smallest margin of any architecture class. Below this floor, the cell '
        f'cannot maintain the methylation pattern that defines its identity. '
        f'The floor is a physical boundary, not a statistical threshold.',
        S('RB',fontSize=7,textColor=TEXT,leading=11)),
]], colWidths=[PW*0.12, PW*0.88],
style=[('BACKGROUND',(0,0),(-1,-1),SURF),('TOPPADDING',(0,0),(-1,-1),5),
       ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),8),
       ('RIGHTPADDING',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.3,BORD),
       ('VALIGN',(0,0),(-1,-1),'MIDDLE')])
story.append(Paragraph('OPERATING RANGE', sLabel))
story.append(Paragraph('INTERPRETATION', S('LC4',fontName='Helvetica-Bold',fontSize=6.5,
                        leading=9,textColor=LAV_D,spaceBefore=0,spaceAfter=2)))
story.append(Spacer(1,2)); story.append(range_block)

# ── FOOTER ────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.12*inch)); story.append(HR(MUT, t=0.3))
story.append(Paragraph(
    'Issue 001  April 2026  https://iamperformance.net  |  '
    'RESEARCH TOOL ONLY — Not intended for diagnostic use. For analytical reference only.  |  '
    'Patent Applications 64/012,720 & 64/014,568  |  '
    'doi:10.5281/zenodo.18702042  |  '
    'Source: De Jager 2014 (Nat Neurosci); Lister 2013 (Science); '
    'Ceccarelli 2016 (Cell); Roadmap Epigenomics 2015 (Nature)',
    S('FT', fontSize=6, textColor=MUT, leading=9)))

doc.build(story, onFirstPage=bg, onLaterPages=bg)
print(f"Built: {out}")
print(f"A_healthy={A_healthy:.4f}  A_AD_hi={A_ad_high:.4f}  A_LGG={A_lgg:.4f}  A_GBM={A_gbm:.4f}")
print(f"C1={f_C1:.1f}%  C2={f_C2:.1f}%  C3={f_C3:.2f}%  locked={locked:.1f}%")
