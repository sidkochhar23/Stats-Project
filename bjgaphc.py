"""
bjgaphc.py -- core routines for two Weibull populations (common shape alpha,
scales beta1, beta2) under balanced joint generalized adaptive progressive
hybrid censoring (BJGAPHC).

Parametrisation (as in the paper):  S_j(t) = exp{-(t/beta_j)^alpha}.

Contents
  1. Censoring / data generation   : censor, simulate_one, thresholds
  2. Sufficient statistics         : pack, take
  3. Classical inference           : mle (unrestricted / order restricted),
                                     obs_info, aci
  4. Bayesian inference            : run_chain (M-H within Gibbs, log scale),
                                     summarise_draws, ess, rhat
Everything in sections 3-4 is vectorised over a batch of independent data sets
(first array dimension B) so that thousands of Monte-Carlo replicates can be
processed at once.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

Z975 = 1.959963984540054


# ----------------------------------------------------------------------------
# 1. Censoring scheme
# ----------------------------------------------------------------------------
def censor(x1, x2, m, R, T1, T2, rng):
    """Apply the balanced joint generalized adaptive progressive hybrid
    censoring scheme (BJGAPHC) to complete lifetimes x1 (Pop1), x2 (Pop2).

    Pre-fixed: m < min(n1,n2); 0 < T1 < T2; R = (R_1,...,R_{m-1}) with
    sum_{i<m}(R_i+1) < min(n1,n2)  (R=None means all R_i = 0).
    Balanced removal (Mondal & Kundu 2019; Shi & Gui 2024): at the i-th failure,
    i<m, if it comes from Pop j then R_i survivors of Pop j and R_i+1 survivors of
    the OTHER population are withdrawn at random, so both populations lose R_i+1
    units per failure.  Generalised adaptive rule (Lee & Lee 2020; Prakash et al.):
    withdrawals are made only at failures occurring before T1; afterwards none.
      case I  : t_m <  T1       -> all survivors withdrawn at t_m, T* = t_m
      case II : T1 <= t_m <= T2 -> no withdrawals between T1 and t_m; all survivors
                                   withdrawn at t_m, T* = t_m
      case III: t_m >  T2       -> stop at T2 (J < m), survivors censored at T2
    Returns the data in the notation of Section 2 of the paper."""
    assert T1 < T2
    n1, n2 = len(x1), len(x2)
    N = n1 + n2
    x = np.concatenate([x1, x2])
    lab = np.concatenate([np.ones(n1, dtype=int), np.zeros(n2, dtype=int)])
    o = np.argsort(x, kind="stable")
    x, lab = x[o], lab[o]
    alive = np.ones(N, dtype=bool)
    ts, ws, r1, r2 = [], [], [], []
    ptr, i, case, Tstar = 0, 0, 0, None

    def withdraw(pop, r):
        idx = np.flatnonzero(alive & (lab == pop))
        r = min(int(r), len(idx))
        if r > 0:
            alive[rng.choice(idx, size=r, replace=False)] = False
        return r

    while True:
        while ptr < N and not alive[ptr]:
            ptr += 1
        if ptr >= N:                                   # nobody left on test
            Tstar = ts[-1] if ts else 0.0
            break
        t = x[ptr]
        if t > T2:                                     # case III
            case, Tstar = 3, T2
            break
        alive[ptr] = False
        ts.append(t); ws.append(lab[ptr]); r1.append(0); r2.append(0)
        i += 1; ptr += 1
        if i < m:
            if t < T1:                                 # adaptive rule: withdrawals only before T1
                Ri = 0 if R is None else int(R[i - 1])
                own = int(lab[ptr - 1])                # population of the failed unit (1 = Pop1)
                a_own = withdraw(own, Ri)
                a_oth = withdraw(1 - own, Ri + 1)
                r1[-1] = a_own if own == 1 else a_oth
                r2[-1] = a_oth if own == 1 else a_own
        else:                                          # m-th failure: stop
            case = 1 if t < T1 else 2
            Tstar = t
            r1[-1] = int((alive & (lab == 1)).sum())
            r2[-1] = int((alive & (lab == 0)).sum())
            alive[:] = False
            break
    K1 = int((alive & (lab == 1)).sum())
    K2 = int((alive & (lab == 0)).sum())
    w = np.array(ws, dtype=float)
    return dict(t=np.array(ts), w=w, R1=np.array(r1, float), R2=np.array(r2, float),
                K1=K1, K2=K2, Tstar=float(Tstar), J=len(ts),
                k1=int(w.sum()), k2=int(len(ts) - w.sum()), case=case)


def simulate_one(rng, n1, n2, alpha, b1, b2, m, R, T1, T2, min_fail=2):
    """One BJGAPHC sample; re-drawn until k1,k2 >= min_fail (MLE exists)."""
    while True:
        x1 = b1 * rng.weibull(alpha, n1)
        x2 = b2 * rng.weibull(alpha, n2)
        d = censor(x1, x2, m, R, T1, T2, rng)
        if d["k1"] >= min_fail and d["k2"] >= min_fail:
            return d


# (expected quantile of the m-th failure time of the balanced progressive Type-II
#  experiment without thresholds) at which (T1, T2) are placed
BRANCH_LEVELS = {"C1": (0.95, 0.99),    # mostly case I   (t_m < T1)
                 "C2": (0.15, 0.85),    # mostly case II  (T1 <= t_m <= T2)
                 "C3": (0.05, 0.30)}    # mostly case III (t_m > T2)


def thresholds(branch, n1, n2, alpha, b1, b2, m, R, rng, reps=1500):
    """(T1,T2) as quantiles of t_m in the experiment WITHOUT thresholds (pilot
    simulation under the given parameters), so that the three termination cases
    of the scheme all occur in the study.  Empirical case frequencies are reported."""
    tm = np.array([simulate_one(rng, n1, n2, alpha, b1, b2, m, R, 1e9, 2e9, min_fail=0)["Tstar"]
                   for _ in range(reps)])
    p1, p2 = BRANCH_LEVELS[branch]
    return float(np.quantile(tm, p1)), float(np.quantile(tm, p2))


def removal_plan(scheme, nmin, m):
    """R = (R_1..R_{m-1}).  Scheme I: all R_i = 0 (at every failure one survivor of
    the other population is withdrawn).  Scheme II: additionally R_i = (nmin-m)//2
    at the floor(m/3)-th failure.  Always sum_{i<m}(R_i+1) < nmin."""
    R = np.zeros(m - 1, dtype=int)
    if scheme == "II":
        R[max(m // 3, 1) - 1] = (nmin - m) // 2
    assert (R + 1).sum() < nmin, "need sum_{i<m}(R_i+1) < min(n1,n2)"
    return R


# ----------------------------------------------------------------------------
# 2. Sufficient statistics, padded to a batch
# ----------------------------------------------------------------------------
def pack(ds):
    B = len(ds)
    Jm = max(d["J"] for d in ds)
    t = np.ones((B, Jm)); c1 = np.zeros((B, Jm)); c2 = np.zeros((B, Jm))
    for b, d in enumerate(ds):
        J = d["J"]
        t[b, :J] = d["t"]
        c1[b, :J] = d["R1"] + d["w"]            # (R_1i + w_i)
        c2[b, :J] = d["R2"] + 1 - d["w"]        # (R_2i + 1 - w_i)
    lt = np.log(t)
    f = lambda k: np.array([d[k] for d in ds], float)
    return dict(lt=lt, c1=c1, c2=c2, K1=f("K1"), K2=f("K2"),
                lTs=np.log(f("Tstar")), J=f("J"), k1=f("k1"), k2=f("k2"),
                sumlt=lt.sum(1))


def take(S, idx):
    return {k: v[idx] for k, v in S.items()}


def U_batch(alpha, S):
    """U1(alpha), U2(alpha) of eq. (5); alpha has shape (B,)."""
    E = np.exp(alpha[:, None] * S["lt"])
    T = np.exp(alpha * S["lTs"])
    return (S["c1"] * E).sum(1) + S["K1"] * T, (S["c2"] * E).sum(1) + S["K2"] * T


def U_grid(G, S):
    """U1, U2 on a common grid G (shape (G,)) -> arrays (B, G)."""
    E = np.exp(G[None, :, None] * S["lt"][:, None, :])
    T = np.exp(G[None, :] * S["lTs"][:, None])
    U1 = np.einsum("bj,bgj->bg", S["c1"], E) + S["K1"][:, None] * T
    U2 = np.einsum("bj,bgj->bg", S["c2"], E) + S["K2"][:, None] * T
    return U1, U2


def U_deriv(alpha, S):
    a = alpha[:, None]
    E = np.exp(a * S["lt"])
    T = np.exp(alpha * S["lTs"])
    out = []
    for c, K in ((S["c1"], S["K1"]), (S["c2"], S["K2"])):
        cE = c * E
        U = cE.sum(1) + K * T
        d1 = (cE * S["lt"]).sum(1) + K * T * S["lTs"]
        d2 = (cE * S["lt"] ** 2).sum(1) + K * T * S["lTs"] ** 2
        out.append((U, d1, d2))
    return out


# ----------------------------------------------------------------------------
# 3. Classical inference
# ----------------------------------------------------------------------------
def _bc(v, a):
    return v.reshape((-1,) + (1,) * (a.ndim - 1))


def profile_from_U(a, U1, U2, S, restricted=False):
    """Profile log-likelihood of alpha: eq. (11) (unrestricted) or the
    order-restricted version built from eqs. (13)-(14)."""
    J, k1, k2, sl = _bc(S["J"], a), _bc(S["k1"], a), _bc(S["k2"], a), _bc(S["sumlt"], a)
    base = J * np.log(a) + (a - 1) * sl - J
    unres = base - k1 * np.log(U1 / k1) - k2 * np.log(U2 / k2)
    if not restricted:
        return unres
    pooled = base - J * np.log((U1 + U2) / J)
    return np.where(U1 / k1 > U2 / k2, pooled, unres)


def _mle_block(S, restricted, ngrid, lo, hi, iters):
    G = np.exp(np.linspace(np.log(lo), np.log(hi), ngrid))
    U1, U2 = U_grid(G, S)
    L = profile_from_U(np.broadcast_to(G, U1.shape), U1, U2, S, restricted)
    ig = L.argmax(1)
    a = np.log(G[np.clip(ig - 1, 0, ngrid - 1)])
    b = np.log(G[np.clip(ig + 1, 0, ngrid - 1)])
    gr = (np.sqrt(5) - 1) / 2

    def f(la):
        al = np.exp(la)
        u1, u2 = U_batch(al, S)
        return profile_from_U(al, u1, u2, S, restricted)

    c = b - gr * (b - a); d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        left = fc > fd
        b = np.where(left, d, b); a = np.where(left, a, c)
        c = b - gr * (b - a); d = a + gr * (b - a)
        fc, fd = f(c), f(d)
    alpha = np.exp(0.5 * (a + b))
    U1, U2 = U_batch(alpha, S)
    r1, r2 = U1 / S["k1"], U2 / S["k2"]
    be1, be2 = r1 ** (1 / alpha), r2 ** (1 / alpha)
    if restricted:
        viol = r1 > r2
        pooled = ((U1 + U2) / S["J"]) ** (1 / alpha)
        be1 = np.where(viol, pooled, be1); be2 = np.where(viol, pooled, be2)
    ll = profile_from_U(alpha, U1, U2, S, restricted)
    return alpha, be1, be2, ll


def mle(S, restricted=False, ngrid=90, lo=0.05, hi=60.0, iters=60, block=200):
    """(alpha_hat, beta1_hat, beta2_hat, max log-lik) for a batch.
    Maximises the profile log-likelihood by a grid search + golden section
    (derivative-free, robust); restricted=True imposes beta1 <= beta2."""
    B = len(S["J"])
    outs = [_mle_block(take(S, slice(s, s + block)), restricted, ngrid, lo, hi, iters)
            for s in range(0, B, block)]
    return tuple(np.concatenate(z) for z in zip(*outs))


def obs_info(alpha, be1, be2, S):
    """Observed information matrix, minus the Hessian of eq. (6); shape (B,3,3)."""
    (U1, d1U1, d2U1), (U2, d1U2, d2U2) = U_deriv(alpha, S)
    J, k1, k2 = S["J"], S["k1"], S["k2"]
    l1, l2 = np.log(be1), np.log(be2)
    a = alpha
    H = np.zeros((len(a), 3, 3))
    H[:, 0, 0] = -J / a ** 2 + sum(
        be ** (-a) * (2 * l * dU - l ** 2 * U - d2U)
        for be, l, U, dU, d2U in ((be1, l1, U1, d1U1, d2U1), (be2, l2, U2, d1U2, d2U2)))
    H[:, 1, 1] = a * k1 / be1 ** 2 - a * (a + 1) * U1 / be1 ** (a + 2)
    H[:, 2, 2] = a * k2 / be2 ** 2 - a * (a + 1) * U2 / be2 ** (a + 2)
    H[:, 0, 1] = H[:, 1, 0] = -k1 / be1 + be1 ** (-a - 1) * (U1 + a * d1U1 - a * U1 * l1)
    H[:, 0, 2] = H[:, 2, 0] = -k2 / be2 + be2 ** (-a - 1) * (U2 + a * d1U2 - a * U2 * l2)
    return -H


def aci(alpha, be1, be2, S, z=Z975):
    """Approximate (Wald) 95% CIs, eq. (16). Returns (B,3,2); NaN if singular."""
    I = obs_info(alpha, be1, be2, S)
    th = np.stack([alpha, be1, be2], 1)
    out = np.full((len(alpha), 3, 2), np.nan)
    for b in range(len(alpha)):
        try:
            v = np.diag(np.linalg.inv(I[b]))
            if np.all(v > 0):
                se = np.sqrt(v)
                out[b, :, 0] = th[b] - z * se; out[b, :, 1] = th[b] + z * se
        except np.linalg.LinAlgError:
            pass
    return out


def bootstrap_p(rng, est, n1, n2, m, Rplan, T1, T2, B, gamma=0.05):
    """Parametric bootstrap-p interval (eq. 17) for ONE data set; the bootstrap
    samples are censored by exactly the same BJGAPHC rule. Returns (3,2)."""
    ds = [simulate_one(rng, n1, n2, est[0], est[1], est[2], m, Rplan, T1, T2)
          for _ in range(B)]
    a, b1, b2, _ = mle(pack(ds))
    q = np.quantile(np.stack([a, b1, b2], 1), [gamma / 2, 1 - gamma / 2], axis=0)
    return q.T


# ----------------------------------------------------------------------------
# 4. Bayesian inference
# ----------------------------------------------------------------------------
HYP_NONINF = dict(a0=1e-3, b0=1e-3, a1=1e-3, b1=1e-3, a2=1e-3, b2=1e-3,
                  g1=1e-3, h1=1e-3, g2=1e-3, h2=1e-3)


def _logprior_beta(l1, l2, prior, hyp, restricted):
    """log prior of (beta1, beta2) (up to a constant) at log-scale values.
    prior='joint': BG (eq. 19) / OBG (eq. 30);  prior='indep': independent
    Gamma(g_j, h_j) priors (truncated to beta1<=beta2 when restricted)."""
    be1, be2 = np.exp(l1), np.exp(l2)
    if prior == "joint":
        a1, bb1, a2, bb2 = hyp["a1"], hyp["b1"], hyp["a2"], hyp["b2"]
        s = be1 + be2
        tail = (a1 - a2 - bb2) * np.log(s) - bb1 * s
        if restricted:
            core = np.logaddexp((a2 - 1) * l1 + (bb2 - 1) * l2,
                                (bb2 - 1) * l1 + (a2 - 1) * l2)
        else:
            core = (a2 - 1) * l1 + (bb2 - 1) * l2
        lp = core + tail
    else:
        lp = ((hyp["g1"] - 1) * l1 - hyp["h1"] * be1
              + (hyp["g2"] - 1) * l2 - hyp["h2"] * be2)
    if restricted:
        lp = np.where(l1 <= l2, lp, -np.inf)
    return lp


def run_chain(S, init, prior, restricted, hyp, rng, N=12000, burn=2000, thin=2,
              target=0.35, adapt_every=25):
    """Metropolis-Hastings within Gibbs (paper Sec. 3.1.2) for a batch of data
    sets.  Each sweep: (i) update alpha | beta1,beta2; (ii) update (beta1,beta2)
    | alpha.  Random-walk proposals are made on the log scale (so they always
    stay in (0,inf) and the proposal is symmetric on that scale; the target
    includes the Jacobian).  Proposal scales are tuned during burn-in only.
    Returns draws (B, M, 3) [alpha, beta1, beta2], acceptance rates (B,2)."""
    B = len(S["J"])
    J, k1, k2, sl = S["J"], S["k1"], S["k2"], S["sumlt"]
    al = np.array(init[0], float); la = np.log(al)
    l1 = np.log(np.array(init[1], float)); l2 = np.log(np.array(init[2], float))
    if restricted:
        l1 = np.minimum(l1, l2)
    U1, U2 = U_batch(al, S)
    sa = np.full(B, 0.2); sb = np.full(B, 0.2)

    def f_alpha(la_, al_, U1_, U2_, l1_, l2_):
        return ((J + hyp["a0"]) * la_ - hyp["b0"] * al_ + (al_ - 1) * sl
                - al_ * (k1 * l1_ + k2 * l2_)
                - U1_ * np.exp(-al_ * l1_) - U2_ * np.exp(-al_ * l2_))

    def f_beta(l1_, l2_, al_, U1_, U2_):
        return (_logprior_beta(l1_, l2_, prior, hyp, restricted) + l1_ + l2_
                - al_ * (k1 * l1_ + k2 * l2_)
                - U1_ * np.exp(-al_ * l1_) - U2_ * np.exp(-al_ * l2_))

    M = (N - burn) // thin
    D = np.empty((B, M, 3), dtype=np.float32)
    win_a = np.zeros(B); win_b = np.zeros(B); post_a = np.zeros(B); post_b = np.zeros(B)
    keep = 0
    with np.errstate(all="ignore"):
        for s in range(N):
            fa = f_alpha(la, al, U1, U2, l1, l2)
            la_p = la + sa * rng.standard_normal(B); al_p = np.exp(la_p)
            U1p, U2p = U_batch(al_p, S)
            fa_p = f_alpha(la_p, al_p, U1p, U2p, l1, l2)
            acc = np.log(rng.random(B)) < fa_p - fa
            la = np.where(acc, la_p, la); al = np.where(acc, al_p, al)
            U1 = np.where(acc, U1p, U1); U2 = np.where(acc, U2p, U2)

            fb = f_beta(l1, l2, al, U1, U2)
            l1p = l1 + sb * rng.standard_normal(B); l2p = l2 + sb * rng.standard_normal(B)
            fb_p = f_beta(l1p, l2p, al, U1, U2)
            accb = np.log(rng.random(B)) < fb_p - fb
            l1 = np.where(accb, l1p, l1); l2 = np.where(accb, l2p, l2)

            if s < burn:
                win_a += acc; win_b += accb
                if (s + 1) % adapt_every == 0:
                    sa = np.clip(sa * np.exp(1.5 * (win_a / adapt_every - target)), 1e-3, 3.0)
                    sb = np.clip(sb * np.exp(1.5 * (win_b / adapt_every - target)), 1e-3, 3.0)
                    win_a[:] = 0; win_b[:] = 0
            else:
                post_a += acc; post_b += accb
                if (s - burn) % thin == 0 and keep < M:
                    D[:, keep, 0] = al; D[:, keep, 1] = np.exp(l1); D[:, keep, 2] = np.exp(l2)
                    keep += 1
    n_post = N - burn
    return D, np.stack([post_a / n_post, post_b / n_post], 1)


def summarise_draws(D, c_list=(1.0, -1.0), gamma=0.05):
    """Posterior summaries for draws D (B,M,3).
    Returns dict: sel (B,3), lx{c} (B,3), sci (B,3,2), hpd (B,3,2)."""
    D = D.astype(np.float64)
    B, M, _ = D.shape
    out = {"sel": D.mean(1), "med": np.median(D, axis=1)}
    for c in c_list:
        out[f"lx{c:+g}"] = -(logsumexp(-c * D, axis=1) - np.log(M)) / c
    q = np.quantile(D, [gamma / 2, 1 - gamma / 2], axis=1)           # (2,B,3)
    out["sci"] = np.stack([q[0], q[1]], axis=2)
    Ds = np.sort(D, axis=1)
    k = int(np.floor((1 - gamma) * M))
    w = Ds[:, k:, :] - Ds[:, : M - k, :]
    i = w.argmin(1)                                                   # (B,3)
    lo = np.take_along_axis(Ds, i[:, None, :], 1)[:, 0, :]
    hi = np.take_along_axis(Ds, (i + k)[:, None, :], 1)[:, 0, :]
    out["hpd"] = np.stack([lo, hi], axis=2)
    return out


def ess(x):
    """Effective sample size of one chain (Geyer initial positive sequence)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return float(n)
    ac = ac / ac[0]
    s = 0.0
    for k in range(0, n - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        s += pair
    tau = max(-1 + 2 * s, 1.0)
    return n / tau


def rhat(chains):
    """Gelman-Rubin potential scale reduction; chains shape (nchains, M)."""
    n = chains.shape[1]
    W = chains.var(axis=1, ddof=1).mean()
    Bv = n * chains.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + Bv / n) / W))
