# Code for Section 4 (numerical study)

Paper: *Classical and Bayesian estimation of Weibull distribution under balanced joint generalized adaptive progressive hybrid censoring*.

| file | purpose |
|---|---|
| `bjgaphc.py` | the methods of Sections 2-3: BJGAPHC sampling (`censor`, `simulate_one`), MLE (unrestricted / order-restricted), observed information and approximate intervals, parametric bootstrap-p, the Metropolis-Hastings-within-Gibbs sampler (BG/OBG and independent Gamma priors), SEL / LINEX estimates, equal-tailed and HPD intervals, ESS and R-hat |
| `simulation.py` | Section 4.1: one design (n1 = n2 = 30, m = 18, R_6 = 6), two parameter settings A and B, 1000 replications each |
| `real_data.py` | Section 4.2: OLED data (model checks, censored sample, MLEs, Bayes estimates, prior sensitivity, figure) |
| `make_tables.py` | writes the LaTeX tables and the figure (pgfplots code) of Section 4 from the json results |
| `data/` | the two OLED samples (hours / 1000), from Zhang et al. (2012) as reported by Shi and Gui (2024) |
| `results/` | the results used in the paper |

Requirements: Python >= 3.9 with `numpy`, `scipy`, `matplotlib`.

```bash
python simulation.py --only A --out sim_A.json      # about 2 minutes
python simulation.py --only B --out sim_B.json      # about 2 minutes
python real_data.py --data1 data/oled_12.36mA.txt --data2 data/oled_9.64mA.txt --m 7 --out real_out   # about 4 minutes
python make_tables.py real_out tables               # needs sim_A.json, sim_B.json in the current folder; writes tables/*.tex and tables/fig_real.tex
```
Random numbers are generated from `numpy.random.default_rng` with fixed seeds, so the results are reproducible.
