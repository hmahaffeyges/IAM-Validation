"""Each cell the IAM-Deconvolver found in a sample, read by the chain against IAM-Atlas v2.1 in the sample's own shares (2026-11-07; author:
"the decovolver is ALWAYS the first things we use ... then each one of them is scored using their appropriate cell references from the
atlas's"). Takes only what the one command wrote in its bundle (met_a_each_cell, cscore_each_cell); a cell is "found" exactly when the chain
did not mark it not found (its not_found_flag); no share threshold is chosen here. Development: these readings are not commissioned."""
def cells_found(B):
    M = {c["cell"]: c for c in ((B.get("met_a_each_cell") or {}).get("cells") or []) if not c.get("not_found_flag")}
    C = {c["cell"]: c for c in ((B.get("cscore_each_cell") or {}).get("cells") or [])}
    return [dict(cell=k, share=v.get("share"), met_a=v.get("A"), met_a_se=v.get("se_over_sites"), c_score=(C.get(k) or {}).get("C"),
                 c_limit=(C.get(k) or {}).get("limit"), c_within=(C.get(k) or {}).get("within")) for k, v in M.items()]
