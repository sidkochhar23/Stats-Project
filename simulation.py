"""
simulation.py -- Monte-Carlo study of Section 4.1.

One design:  n1 = n2 = 30, m = 18 failures, withdrawal plan R_6 = 6 (all other R_i = 0),
thresholds T1 < T2 fixed per setting (placed so that the test usually ends in case II/III).
Two settings (alpha, beta1, beta2):
    A (well separated): (1.5, 1.2, 1.8)        B (equal scales): (1.5, 1.5, 1.5)
For every setting B = 1000 BJGAPHC samples are generated and analysed by
  * MLE, unrestricted and order-restricted (beta1 <= beta2)
  * 95% approximate (Fisher) and bootstrap-p intervals
  * Bayes (SEL and LINEX c = 1) with the joint BG prior (unrestricted) / OBG prior (restricted)
    and 95% HPD intervals
  * prior sensitivity: unrestricted Bayes (SEL) with independent Gamma(a,b) priors on
    alpha, beta1, beta2 for five choices of (a,b).
Usage:  python simulation.py [--reps 1000] [--B 200] [--out sim_results.json]
"""
import argparse, json, time
import numpy as np
import bjgaphc as bj

SETTINGS = {"A": (1.5, 1.2, 1.8), "B": (1.5, 1.5, 1.5)}
N1 = N2 = 30
M = 18
R = np.zeros(M - 1, dtype=int); R[5] = 6          # R_6 = 6, all other R_i = 0
assert (R + 1).sum() < min(N1, N2)

# main Bayes prior: prior means equal to the true alpha (1.5) and true beta1+beta2 (3);
# beta1/(beta1+beta2) ~ Beta(2,3) (mean 0.4)
HYP_MAIN = dict(a0=3.0, b0=2.0, a1=3.0, b1=1.0, a2=2.0, b2=3.0, g1=1, h1=1, g2=1, h2=1)
PRIORS = [(0.1, 0.1), (1, 1), (2, 1), (4, 2), (8, 2)]      # Gamma(a, b) for alpha, beta1, beta2


def hyp_gamma(a, b):
    return dict(a0=a, b0=b, g1=a, h1=b, g2=a, h2=b, a1=a, b1=b, a2=a, b2=b)


def run_setting(name, reps, Bboot, seed, N=10000, burn=2000, thin=2):
    th = np.array(SETTINGS[name])
    rng = np.random.default_rng(np.random.SeedSequence([seed, ord(name)]))
    # thresholds: 15% and 85% quantiles of t_m in the experiment without thresholds (pilot)
    T1, T2 = bj.thresholds("C2", N1, N2, *th, M, R, rng)
    ds = [bj.simulate_one(rng, N1, N2, *th, M, R, T1, T2) for _ in range(reps)]
    cases = np.array([d["case"] for d in ds])
    S = bj.pack(ds)
    out = dict(setting=name, truth=th.tolist(), T1=T1, T2=T2, reps=reps, B=Bboot,
               case_freq=[float((cases == c).mean()) for c in (1, 2, 3)],
               mean_J=float(np.mean([d["J"] for d in ds])),
               mean_k=[float(np.mean([d["k1"] for d in ds])), float(np.mean([d["k2"] for d in ds]))])
    a, e1, e2, _ = bj.mle(S, False)
    ar, r1, r2, _ = bj.mle(S, True)
    U = np.stack([a, e1, e2], 1); Rr = np.stack([ar, r1, r2], 1)
    out["pooled_freq"] = float(np.mean(e1 > e2))           # unrestricted estimate violates the order

    def stats(x):
        return dict(bias=(x - th).mean(0).tolist(), mse=((x - th) ** 2).mean(0).tolist())

    out["mle_u"], out["mle_r"] = stats(U), stats(Rr)
    d = (Rr - th) ** 2 - (U - th) ** 2                     # paired difference in squared error
    out["mle_dmse"] = dict(mean=d.mean(0).tolist(), se=(d.std(0, ddof=1) / np.sqrt(reps)).tolist())

    def cover(ci):
        ok = ~np.isnan(ci).any(axis=(1, 2))
        return dict(cp=((ci[ok][..., 0] <= th) & (ci[ok][..., 1] >= th)).mean(0).tolist(),
                    al=(ci[ok][..., 1] - ci[ok][..., 0]).mean(0).tolist())

    out["aci"] = cover(bj.aci(a, e1, e2, S))
    boot = np.empty((reps, 3, 2))
    for r in range(reps):
        boot[r] = bj.bootstrap_p(rng, (a[r], e1[r], e2[r]), N1, N2, M, R, T1, T2, Bboot)
    out["boot"] = cover(boot)

    # Bayes, main prior: unrestricted (BG) and restricted (OBG)
    sel, lx, hpd, accept, essl = {}, {}, {}, {}, {}
    for lab, init, restr, prior in (("u", U, False, "joint"), ("r", Rr, True, "joint")):
        D, acc = bj.run_chain(S, tuple(init.T), prior, restr, HYP_MAIN, rng, N=N, burn=burn, thin=thin)
        sm = bj.summarise_draws(D, c_list=(1.0,))
        sel[lab], lx[lab], hpd[lab] = sm["sel"], sm["lx+1"], sm["hpd"]
        accept[lab] = acc.mean(0).tolist()
        essl[lab] = [float(np.median([bj.ess(D[i, :, k]) for i in range(20)])) for k in range(3)]
        if lab == "u":
            post_order = float(np.mean(D[:, :, 1] <= D[:, :, 2]))   # P(beta1 <= beta2 | data), averaged
    out["bayes_sel_u"], out["bayes_sel_r"] = stats(sel["u"]), stats(sel["r"])
    out["bayes_lx_u"], out["bayes_lx_r"] = stats(lx["u"]), stats(lx["r"])
    out["hpd"] = cover(hpd["u"])
    out["post_prob_order"] = post_order
    out["mcmc"] = dict(acc=accept, ess=essl)

    # prior sensitivity (unrestricted, independent Gamma(a,b) priors, SEL)
    out["sens"] = {}
    for (pa, pb) in PRIORS:
        D, _ = bj.run_chain(S, tuple(U.T), "indep", False, hyp_gamma(pa, pb), rng, N=N, burn=burn, thin=thin)
        out["sens"][f"{pa},{pb}"] = stats(bj.summarise_draws(D, c_list=())["sel"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--B", type=int, default=200, help="bootstrap resamples per replication")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", default="", help="run only this setting (A or B)")
    ap.add_argument("--out", default="sim_results.json")
    a = ap.parse_args()
    res = {}
    for name in SETTINGS:
        if a.only and a.only != name:
            continue
        t0 = time.time()
        res[name] = run_setting(name, a.reps, a.B, a.seed)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"setting {name} done in {time.time()-t0:.0f}s", flush=True)
