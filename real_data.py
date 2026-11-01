"""
real_data.py -- real-data illustration of Section 4.2 (one data set, one censoring design).

Usage:
  python real_data.py --data1 data/oled_12.36mA.txt --data2 data/oled_9.64mA.txt --m 7 --out real_out
data1 = Pop1 (the population expected to have the SHORTER lifetimes, so that beta1 <= beta2).
Steps: Weibull fits and parametric-bootstrap Kolmogorov-Smirnov p-values, likelihood-ratio test of
a common shape, BJGAPHC applied to the complete data (R_i = 0, thresholds = 15% / 85% quantiles of
the m-th failure time under the model fitted to the complete data), MLEs (unrestricted /
restricted) with approximate and bootstrap-p intervals, Bayes estimates (joint BG / OBG prior,
posterior mean and 95% HPD, four over-dispersed chains) and a prior-sensitivity table.
Writes real_results.json to --out.
"""
import argparse, json, os, re
import numpy as np
from scipy import stats
import bjgaphc as bj


def read(path):
    return np.array([float(v) for v in re.split(r"[,\s;]+", open(path).read().strip()) if v])


def fit1(x):
    c, _, sc = stats.weibull_min.fit(x, floc=0)
    return c, sc, float(stats.weibull_min.logpdf(x, c, 0, sc).sum())


def ks_boot(x, B, rng):
    c, sc, _ = fit1(x)
    D = stats.kstest(x, "weibull_min", args=(c, 0, sc)).statistic
    cnt = 0
    for _ in range(B):
        y = sc * rng.weibull(c, len(x))
        cy, sy, _ = fit1(y)
        cnt += stats.kstest(y, "weibull_min", args=(cy, 0, sy)).statistic >= D
    return D, (cnt + 1) / (B + 1)


def chains(S, est, prior, restricted, hyp, rng, N, burn, thin):
    """4 over-dispersed chains for one data set; returns pooled draws, R-hat, ESS, acceptance."""
    S4 = bj.take(S, [0, 0, 0, 0])
    init = [est[k] * np.exp(0.25 * rng.standard_normal(4)) for k in range(3)]
    D, acc = bj.run_chain(S4, init, prior, restricted, hyp, rng, N=N, burn=burn, thin=thin)
    rh = [bj.rhat(D[:, :, k].astype(float)) for k in range(3)]
    es = [float(sum(bj.ess(D[c, :, k]) for c in range(4))) for k in range(3)]
    return D.reshape(1, -1, 3), rh, es, acc.mean(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data1", required=True); ap.add_argument("--data2", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--burn", type=int, default=5000)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="real_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    x1, x2 = read(a.data1), read(a.data2)
    n1, n2 = len(x1), len(x2)
    res = dict(n=[n1, n2], m=a.m)

    # ---- model validation ---------------------------------------------------------------
    f1, f2 = fit1(x1), fit1(x2)
    ks = [ks_boot(x1, 1000, rng), ks_boot(x2, 1000, rng)]
    N = n1 + n2
    Sfull = bj.pack([bj.censor(x1, x2, N, None, 0.0, np.inf, rng)])
    af, b1f, b2f, llj = bj.mle(Sfull)
    full = np.array([af[0], b1f[0], b2f[0]])
    ll_sep = f1[2] + f2[2]
    lrt = 2 * (ll_sep - float(llj[0]))
    res["fit"] = dict(alpha_sep=[f1[0], f2[0]], beta_sep=[f1[1], f2[1]], ks_D=[ks[0][0], ks[1][0]],
                      ks_p=[ks[0][1], ks[1][1]], full_mle=full.tolist(), lrt=lrt,
                      lrt_p=float(stats.chi2.sf(max(lrt, 0), 1)),
                      aic=[6 - 2 * float(llj[0]), 8 - 2 * ll_sep],
                      bic=[3 * np.log(N) - 2 * float(llj[0]), 4 * np.log(N) - 2 * ll_sep])

    # ---- censoring ------------------------------------------------------------------------
    R = bj.removal_plan("I", min(n1, n2), a.m)
    T1, T2 = bj.thresholds("C2", n1, n2, *full, a.m, R, rng)
    # the units withdrawn are chosen at random; as in the simulations we condition on k1, k2 >= 2
    # (the MLE needs failures from both populations) by re-drawing the random withdrawals
    for attempt in range(1, 201):
        d = bj.censor(x1, x2, a.m, R, T1, T2, rng)
        if d["k1"] >= 2 and d["k2"] >= 2:
            break
    else:
        raise SystemExit("no admissible censored sample found; change --m")
    res["attempts"] = attempt
    S = bj.pack([d])
    res["design"] = dict(m=a.m, T1=T1, T2=T2, case=d["case"], J=d["J"], k1=d["k1"], k2=d["k2"],
                         K1=d["K1"], K2=d["K2"], Tstar=d["Tstar"],
                         t=d["t"].tolist(), pop=[2 - int(w) for w in d["w"]],
                         R1=d["R1"].tolist(), R2=d["R2"].tolist())
    # ---- MLEs -----------------------------------------------------------------------------
    au, u1, u2, _ = bj.mle(S, False); ar, r1, r2, _ = bj.mle(S, True)
    est_u = np.array([au[0], u1[0], u2[0]]); est_r = np.array([ar[0], r1[0], r2[0]])
    boot = bj.bootstrap_p(rng, est_u, n1, n2, a.m, R, T1, T2, a.B)
    res["mle"] = dict(unrestricted=est_u.tolist(), restricted=est_r.tolist(),
                      aci=bj.aci(au, u1, u2, S)[0].tolist(), boot=boot.tolist())

    # ---- Bayes: joint BG (unrestricted) / OBG (restricted), hyper-parameters 0.1 ---------------
    hyp = dict(a0=.1, b0=.1, a1=.1, b1=.1, a2=.1, b2=.1, g1=.1, h1=.1, g2=.1, h2=.1)
    res["bayes"] = {}
    for lab, est, restr in (("unrestricted", est_u, False), ("restricted", est_r, True)):
        D, rh, es, acc = chains(S, est, "joint", restr, hyp, rng, a.iters, a.burn, a.thin)
        sm = bj.summarise_draws(D, c_list=(1.0,))
        if lab == "unrestricted":
            draws_u = D[0]
        res["bayes"][lab] = dict(mean=sm["sel"][0].tolist(), linex1=sm["lx+1"][0].tolist(),
                                 hpd=sm["hpd"][0].tolist(), rhat=rh, ess=es, acc=acc,
                                 post_order=float(np.mean(D[0, :, 1] <= D[0, :, 2])))
    # ---- prior sensitivity (independent Gamma(a,b) for alpha, beta1, beta2; unrestricted) ----
    res["sens"] = {}
    for (pa, pb) in [(1, 1), (2, 1), (0.1, 0.1), (1, 0.1)]:
        h = dict(a0=pa, b0=pb, g1=pa, h1=pb, g2=pa, h2=pb, a1=pa, b1=pb, a2=pa, b2=pb)
        D, rh, es, acc = chains(S, est_u, "indep", False, h, rng, a.iters, a.burn, a.thin)
        res["sens"][f"{pa},{pb}"] = dict(mean=bj.summarise_draws(D, c_list=())["sel"][0].tolist(),
                                         post_order=float(np.mean(D[0, :, 1] <= D[0, :, 2])),
                                         rhat=max(rh))

    # ---- figure: profile log-likelihood of alpha and posterior histograms --------------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    grid = np.linspace(2.0, 25.0, 300)
    U1g, U2g = bj.U_grid(grid, S)
    prof = bj.profile_from_U(np.broadcast_to(grid, U1g.shape), U1g, U2g, S, False)[0]
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.8))
    ax[0].plot(grid, prof - prof.max(), color="k"); ax[0].axvline(est_u[0], color="r", ls="--")
    ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel("profile log-likelihood (relative)")
    for k, (lab, xl) in enumerate([(r"$\alpha$", 0), (r"$\beta_1$", 1), (r"$\beta_2$", 2)]):
        ax[k + 1].hist(draws_u[:, xl], bins=60, density=True, color="lightgray", edgecolor="gray")
        ax[k + 1].axvline(est_u[xl], color="r", ls="--", label="MLE")
        ax[k + 1].axvline(draws_u[:, xl].mean(), color="b", label="posterior mean")
        ax[k + 1].set_xlabel(lab)
        if k == 0:
            ax[k + 1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "real_figure.pdf")); plt.close(fig)
    # data behind the figure (also written to the json file so that it can be redrawn in LaTeX)
    fig_data = dict(profile=[[float(g), float(p - prof.max())] for g, p in zip(grid[::6], prof[::6])],
                    mle=est_u.tolist(), mean=draws_u.mean(0).tolist(), hist={})
    for k, nm in enumerate(["alpha", "beta1", "beta2"]):
        dens, edges = np.histogram(draws_u[:, k], bins=30, density=True)
        fig_data["hist"][nm] = dict(edges=edges.tolist(), dens=dens.tolist())
    res["fig"] = fig_data
    json.dump(res, open(os.path.join(a.out, "real_results.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("fit", "design")}, indent=1))
    print("MLE", res["mle"])
    print("Bayes", {k: (v["mean"], v["hpd"], v["rhat"], v["ess"], v["post_order"]) for k, v in res["bayes"].items()})
    print("sens", res["sens"])


if __name__ == "__main__":
    main()
