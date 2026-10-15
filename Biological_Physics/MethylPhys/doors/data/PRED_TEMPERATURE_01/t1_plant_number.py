"""PRED-TEMPERATURE-01 stage T1: the law's number for GSE54292 (Arabidopsis grown for life at 10 C and 16 C). Nothing fitted.
The law: ln[(1 - eps)/eps] scales as 1/T, so colder over warmer log-odds = T_warm / T_cold (kelvin). Writes t1_plant_number.json."""
import json, os
T0 = 273.15; cold, warm = 10.0, 16.0
r = (T0 + warm) / (T0 + cold)
json.dump({"cold_C": cold, "warm_C": warm, "log_odds_ratio_cold_over_warm": round(r, 4)}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t1_plant_number.json"), "w"), indent=1)
print(round(r, 4))
