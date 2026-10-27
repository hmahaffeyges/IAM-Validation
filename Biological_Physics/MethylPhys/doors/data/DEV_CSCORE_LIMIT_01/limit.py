"""DEV-CSCORE-LIMIT-01: the C-Score's healthy upper limit, derived (plan 14.12; written 2026-10-25 before any healthy C-Score is scored against it).
The C-Score (stage_metaa_atlas._clustering, unchanged): departures in genome order, block means of w = 10 neighboring sites times sqrt(w), their
variance over the sites' variance (both with divisor n). Split the sites' sum of squares into between blocks (SSB) and within blocks (SSW):
C = (n / nb) SSB / (SSB + SSW). With departures independent from site to site, SSB and SSW are independent chi-squares with nb - 1 and n - nb
degrees of freedom, so C nb / n is Beta((nb - 1) / 2, (n - nb) / 2) exactly. The limit is its one-sided 95 % point, from the reading's own
site count; nothing measured. Run 2 (run 1, disclosed: it took C as chi-square over nb - 1, ignoring that the denominator contains the
numerator, which widens the upper tail by about 1 / sqrt(1 - 1/w); its 60,000-site maps exceeded the limit in 3.35 % instead of 5 %)."""
from scipy.stats import beta
W = 10
def limit(n_sites, q=0.95, w=W):
    nb = int(n_sites) // w; n = nb * w
    return None if nb < 10 else float(n / nb * beta.ppf(q, (nb - 1) / 2, (n - nb) / 2))

def limit_450k(n_sites, term_file=None, q=0.95):
    """2026-10-26: the whole-genome limit on a 450K array: limit(n) times the frozen platform term (limit_04/platform_term_450k.json)."""
    import json, os
    t = json.load(open(term_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "limit_04", "platform_term_450k.json")))
    l = limit(n_sites, q); return None if l is None else l * t["factor"]
