"""Plain-text to LaTeX for register and provenance tables (used by make_app_G.py and make_app_I.py).

The register stores statements in ASCII ("sigma8 0.809 -> 0.800", "tau ~ m^-5", "6.115e-10"). texify() finds the
mathematical tokens with one ordered regular expression, writes each as inline math, and escapes every other character.
Adjacent math tokens are merged, so "$a$$b$" never opens display math.
"""
import re

_SUB = {"TLS": r"\rm TLS", "qp": r"\rm qp", "IAM": r"\rm IAM", "PD": r"\rm PD", "floor": r"\rm floor",
        "free": r"1,\rm free", "crit": r"\rm crit", "core": r"\rm core", "lens": r"\rm lens", "dyn": r"\rm dyn",
        "info": r"\rm info", "sw": r"\rm sw", "ISW": r"\rm ISW", "BH": r"\rm BH", "cp": r"\rm cp"}


def _exp(m):
    man, e = m.group(1), int(m.group(2))
    return rf"10^{{{e}}}" if man in ("1", "1.0") else rf"{man}\times10^{{{e}}}"


# (pattern, replacement or function) - order matters: longer tokens first
_TOK = [
    (r"(?:Table|Section|Chapter)~\\ref\{[^}]+\}", None),                     # pass-through LaTeX from the overrides file
    (r"\\ref\{[^}]+\}", None),
    (r"~?\\cite\{[^}]+\}", None), (r"\\(?:observed|openprob|prediction|calc)\{\}", None),
    (r"(?<![\w.])(\d+(?:\.\d+)?)e([+-]?\d+)\b", _exp),
    (r"\)e([+-]?\d+)\b", lambda m: rf")\times10^{{{int(m.group(1))}}}"),
    (r"\+/-", r"\pm"), (r"<<", r"\ll"), (r"->", r"\to"), (r">=", r"\ge"), (r"<=", r"\le"), (r"!=", r"\ne"),
    (r"<", "<"), (r">", ">"), (r"~", r"\sim"), (r"(?<![\w.)\]/])-(?=\d)", "-"), (r"(?<=/)-(?=\d)", "-"),
    (r"\bDelta[- ]chi2\b", r"\Delta\chi^2"), (r"\bchi2\b", r"\chi^2"),
    (r"\bf ?sigma8\b", r"f\sigma_8"), (r"\bsigma8\b", r"\sigma_8"), (r"\bS8\b", r"S_8"),
    (r"\bsigma\(mu0\)", r"\sigma(\mu_0)"),
    (r"\bsigma_v\b", r"\sigma_v"), (r"\bsigma_crit\b", r"\sigma_{\rm crit}"), (r"\bsigma\^2\b", r"\sigma^2"),
    (r"\bsigma\b", r"\sigma"),
    (r"\bmu0\b", r"\mu_0"), (r"\bSigma0\b", r"\Sigma_0"), (r"\bmu\b", r"\mu"), (r"\bSigma\b", r"\Sigma"),
    (r"\bbeta_gamma/beta_m\b", r"\beta_\gamma/\beta_m"), (r"\bbeta_gamma\b", r"\beta_\gamma"),
    (r"\bbeta_m\b", r"\beta_m"), (r"\bbeta\b", r"\beta"),
    (r"\bw_info\b", r"w_{\rm info}"), (r"\bw0waCDM\b", r"w_0w_a\rm CDM"), (r"\bw0\b", r"w_0"), (r"\bwa\b", r"w_a"),
    (r"\bw(?= ?(?:=|!=|<|>))", "w"),
    (r"\bOmega_b h\^2\b", r"\Omega_bh^2"), (r"\bOmega_b/Omega_m\b", r"\Omega_b/\Omega_m"),
    (r"\bOmega_(m|b|L)\b", lambda m: rf"\Omega_{m.group(1)}"), (r"\bOmega_Lambda\b", r"\Omega_\Lambda"), (r"\bOmega_DE\b", r"\Omega_{\rm DE}"),
    (r"\bLambdasCDM\b", r"\Lambda_s\rm CDM"), (r"\bLCDM\b", r"\Lambda\rm CDM"),
    (r"\bH\^2_LCDM\b", r"H^2_{\Lambda\rm CDM}"), (r"\bH0\^\(2/5\)", r"H_0^{2/5}"), (r"\bH0\^2\b", r"H_0^2"),
    (r"\bH0\b", r"H_0"),
    (r"\bE\(a\)", r"E(a)"), (r"\bE\(eta\)", r"E(\eta)"),
    (r"\bE_G\^3\b", r"E_G^3"), (r"\bE_G\b", r"E_G"), (r"\bE_sw\b", r"E_{\rm sw}"),
    (r"\bk_?B T_j ln ?2\b", r"k_BT_j\ln2"), (r"\bkB\^2\b", r"k_B^2"), (r"\bkB ?T ln ?2\b", r"k_BT\ln2"),
    (r"\bk_?B\b", r"k_B"), (r"\bT_j\b", r"T_j"), (r"\bhbar\b", r"\hbar"), (r"\bln ?2\b", r"\ln2"),
    (r"\bx_qp\b", r"x_{\rm qp}"), (r"\btau_(TLS|qp|IAM|PD)\b", lambda m: rf"\tau_{{{_SUB[m.group(1)]}}}"),
    (r"\btau\b", r"\tau"), (r"\bn_cp\b", r"n_{\rm cp}"), (r"\blambda_L\^3\b", r"\lambda_L^3"),
    (r"\bT1\*", r"T_1^*"), (r"\bT1_free\b", r"T_{1,\rm free}"), (r"\bT1\b", r"T_1"),
    (r"\b10\^\((-?\d+)\)", lambda m: rf"10^{{{m.group(1)}}}"), (r"\b10\^(-?\d+)", lambda m: rf"10^{{{m.group(1)}}}"),
    (r"\b([A-Za-z])\^\((-?\d+(?:/\d+)?)\)", lambda m: rf"{m.group(1)}^{{{m.group(2)}}}"),
    (r"\b([A-Za-z])\^(-?\d+(?:/\d+)?)", lambda m: rf"{m.group(1)}^{{{m.group(2)}}}"),
    (r"\beps_floor\b", r"\varepsilon_{\rm floor}"), (r"\beps\b", r"\varepsilon"), (r"\beta\b", r"\eta"),
    (r"\bomega0\b", r"\omega_0"), (r"\b2sqrt2\b", r"2\sqrt2"), (r"\bsqrt2\b", r"\sqrt2"), (r"\b2pi\b", r"2\pi"),
    (r"\|S\|", r"|S|"), (r"\bQ_?L\b", r"Q_L"),
    (r"\bM_lens/M_dyn\b", r"M_{\rm lens}/M_{\rm dyn}"), (r"\br_core\b", r"r_{\rm core}"), (r"\bM_BH\b", r"M_{\rm BH}"),
    (r"\bC_phiphi\b", r"C_{\phi\phi}"), (r"\bA_ISW\b", r"A_{\rm ISW}"), (r"\bA_IAM\b", r"A_{\rm IAM}"),
    (r"\bmp/me\b", r"m_p/m_e"), (r"\bm_e\b", r"m_e"), (r"\bP_IAM/P_LCDM\b", r"P_{\rm IAM}/P_{\Lambda\rm CDM}"),
    (r"\bPhi\b", r"\Phi"),
]
_RX = re.compile("|".join(f"({p})" for p, _ in _TOK))
_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "$": r"\$", "^": r"\^{}", "~": r"\textasciitilde{}"}


def esc(s):
    """Escape plain text (file names, identifiers) for LaTeX text mode."""
    return "".join(_ESC.get(c, c) for c in s)


def path(p):
    """A repository path in typewriter type; the preamble's \\_ allows a break after each underscore."""
    return r"\texttt{" + esc(p).replace("/", r"/\allowbreak{}") + "}"


def texify(s):
    out, i = [], 0
    for m in _RX.finditer(s):
        out.append(esc(s[i:m.start()]))
        k = next(j for j, g in enumerate(m.groups()) if g is not None)
        # group index -> token index (each token has its own outer group plus inner groups)
        tok_i, g = 0, 0
        for ti, (p, _) in enumerate(_TOK):
            n = re.compile(p).groups + 1
            if g <= k < g + n:
                tok_i = ti
                break
            g += n
        p, rep = _TOK[tok_i]
        if rep is None:
            out.append(m.group(0))
        else:
            mm = re.match(p, m.group(0))
            out.append("$" + (rep(mm) if callable(rep) else rep) + "$")
        i = m.end()
    out.append(esc(s[i:]))
    t = "".join(out)
    while "$$" in t:
        t = t.replace("$$", "")
    return t


def check(t):
    """Static checks on generated LaTeX: balanced braces, even number of unescaped $."""
    depth = 0
    for j, c in enumerate(t):
        if c == "{" and (j == 0 or t[j - 1] != "\\"):
            depth += 1
        elif c == "}" and (j == 0 or t[j - 1] != "\\"):
            depth -= 1
            if depth < 0:
                return "unbalanced }"
    if depth:
        return "unbalanced {"
    if len(re.findall(r"(?<!\\)\$", t)) % 2:
        return "odd $"
    return None
