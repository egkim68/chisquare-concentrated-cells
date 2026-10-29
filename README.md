# Concentrated Associations in Contingency Tables

Code and protocol for the simulation study by Eungi Kim. Requires Python 3 with numpy, scipy, pandas, and matplotlib.

```
python3 theory_checks.py out/theory
python3 identity_check.py
python3 exp1_gate.py 1500 out/exp1 2
python3 exp2_map.py 200 99 out/exp2_map.csv 2
python3 exp3_letters.py 5000 out/exp3_letters.csv
python3 calib_check.py 150 out/calib_check.csv 2
python3 analysis.py out
python3 make_figures.py out figures 600
```

For the mobility illustration, export the data in R with `write.csv(as.data.frame(occupationalStatus), "occupationalStatus.csv")` and run `python3 illustration_mobility.py occupationalStatus.csv out/illustration`.

Seeds are fixed in the scripts. License: MIT.
