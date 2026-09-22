import pandas as pd
def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
    def norm(o):
        if isinstance(o, dict):   return pd.DataFrame([o])
        if isinstance(o, pd.Series): return pd.DataFrame([o])
        return o
    frames = ([self]+[norm(o) for o in other]) if isinstance(other,(list,tuple)) else [self, norm(other)]
    return pd.concat(frames, ignore_index=ignore_index, sort=sort)
def _ser_append(self, other, ignore_index=False, verify_integrity=False):
    others = other if isinstance(other,(list,tuple)) else [other]
    return pd.concat([self]+list(others), ignore_index=ignore_index)
pd.DataFrame.append = _df_append
pd.Series.append = _ser_append
