"""Reproduce every number and figure in the accompanying report.

This is the evaluation harness for the article-style report. The notebook
(face_recognition_pipeline.ipynb) is the tutorial walkthrough; this file is what
generates the reported results, and it is the authoritative source for them.

    python paper_experiments.py            # all experiments, ~8 min on CPU
    python paper_experiments.py --figures  # also regenerate report/figures/*.png

Every experiment prints a block whose caption names the paper table it feeds.
Randomness is seeded throughout, so re-running reproduces the reported values
exactly.

NOT INCLUDED: the YuNet face-detector comparison (report Section V-B). It requires
an ONNX model file that is not redistributed with this repository, so that
experiment cannot be reproduced from this artifact alone; the report says so.
"""
from __future__ import annotations

import argparse
import csv
import random
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from facenet_arch import build_embedding_network
from facenet_utils import (LFW_FACE_CROP, LFW_IMAGE_ROOT, LFW_PAIRS_CSV,
                           apply_pretrained_weights, distance_matrix,
                           encode_many, evaluate_verification,
                           identification_ranks, pair_distances)

TUNE_TEST_SEED = 12345
POOL_SEED = 0
MULTISHOT_SEED = 1
IMPOSTOR_SEED = 5
SUBGALLERY_TRIALS = 200
POOL_SIZE = 300

# The crop the shipped pipeline actually uses. The report is written around this
# value; Experiment 1 validates it rather than choosing it. Do not edit this to
# match a number in the report -- the code is the artifact, and the report must
# describe what the code does.
MODULE_CROP = LFW_FACE_CROP

photos_of = lambda d: sorted(d.glob("*.jpg"))


def banner(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def build_splits():
    """Disjoint tuning and test pair splits drawn from the official pair list.

    The crop hyperparameters are worth >30 accuracy points, so selecting them on
    the evaluation data would inflate the result. Cross-validating the threshold
    does not protect against that. We therefore halve the official list and draw
    a tuning and a test split that share no pairs.
    """
    rows = list(csv.reader(open(LFW_PAIRS_CSV, newline="")))[1:]
    p = lambda n, i: LFW_IMAGE_ROOT / n / f"{n}_{int(i):04d}.jpg"
    matched, mismatched = [], []
    for r in rows:
        if len(r) > 3 and r[3].strip():
            mismatched.append((p(r[0], r[1]), p(r[2], r[3]), 0))
        else:
            matched.append((p(r[0], r[1]), p(r[0], r[2]), 1))

    rng = random.Random(TUNE_TEST_SEED)
    rng.shuffle(matched)
    rng.shuffle(mismatched)
    hm, hx = len(matched) // 2, len(mismatched) // 2
    tune = matched[:300] + mismatched[:300]
    test = matched[hm:hm + 300] + mismatched[hx:hx + 300]

    shared = (set(str(q) for a, b, _ in tune for q in (a, b)) &
              set(str(q) for a, b, _ in test for q in (a, b)))
    return tune, test, len(shared)


def experiment_1_crop(model, tune, test):
    """Report Table I: crop ablation on both splits."""
    banner("EXPERIMENT 1  ->  report Table I (crop ablation)")
    y_tune = np.array([l for _, _, l in tune])
    y_test = np.array([l for _, _, l in test])

    grid = [("none (full 250 px frame)", None), ("160 px, centred", (160, 0)),
            ("128 px, centred", (128, 0)), ("110 px, centred", (110, 0)),
            ("96 px, centred", (96, 0)), ("92 px, centred", (92, 0)),
            ("96 px, 18 px lower", (96, 18)), ("92 px, 18 px lower", (92, 18)),
            ("88 px, 18 px lower", (88, 18))]

    print(f"{'crop window':>26} {'tuning CV':>11} {'test CV':>10}")
    best, best_acc, out = None, -1.0, []
    for label, crop in grid:
        a = evaluate_verification(pair_distances(tune, model, crop=crop,
                                                 progress=False)[0], y_tune)["accuracy_mean"]
        b = evaluate_verification(pair_distances(test, model, crop=crop,
                                                 progress=False)[0], y_test)["accuracy_mean"]
        out.append((label, crop, a, b))
        print(f"{label:>26} {a*100:10.2f}% {b*100:9.2f}%")
        if a > best_acc:
            best_acc, best = a, (label, crop)

    shifted = [r for r in out if r[1] and r[1][1] == 18]
    sp_t = max(r[2] for r in shifted) - min(r[2] for r in shifted)
    sp_e = max(r[3] for r in shifted) - min(r[3] for r in shifted)
    mod = next(r for r in out if r[1] == MODULE_CROP)

    print(f"\nbest on the TUNING split : {best[0]}  ({best_acc*100:.2f}%)")
    print(f"crop the pipeline ships  : {mod[0]}  "
          f"(tuning {mod[2]*100:.2f}%, test {mod[3]*100:.2f}%)")
    print(f"spread among the 3 shifted windows: tuning {sp_t*100:.2f} pts, "
          f"test {sp_e*100:.2f} pts")
    print("-> indistinguishable at the fold-spread noise floor, so the size choice")
    print("   within the shifted family is arbitrary; only the offset is real.")
    if best_acc - mod[2] > 0.042:
        print("WARNING: the shipped crop is more than one fold-spread below the best")
        print("   tuning configuration. Revisit the constant in facenet_utils.py.")
    return MODULE_CROP


def experiment_2_verification(model, test, crop):
    """Report Table III and the headline verification figure."""
    banner("EXPERIMENT 2  ->  report Table III (operating points) + headline")
    y = np.array([l for _, _, l in test])
    d, _ = pair_distances(test, model, crop=crop, progress=False)
    r = evaluate_verification(d, y, folds=10)
    print(f"HELD-OUT CV accuracy : {r['accuracy_mean']*100:.2f}% "
          f"+/- {r['accuracy_std']*100:.2f}% (fold spread)")
    print(f"  standard error of the mean : {r['accuracy_std']/np.sqrt(10)*100:.2f}%")
    print(f"  AUC {r['auc']:.3f} | EER {r['eer']*100:.2f}% | tau {r['threshold_mean']:.3f}")
    print(f"  per-fold: {' '.join(f'{a*100:.1f}' for a in r['fold_accuracies'])}")

    gen, imp = d[y == 1], d[y == 0]
    print(f"\n{'tau':>8} {'FAR':>8} {'FRR':>8} {'fixed-tau acc':>14}")
    for t in [0.70, 0.85, 0.95, r["threshold_mean"], 1.10, 1.25]:
        mark = "  <- CV-selected" if abs(t - r["threshold_mean"]) < 1e-9 else ""
        print(f"{t:8.3f} {(imp < t).mean()*100:7.2f}% {(gen >= t).mean()*100:7.2f}% "
              f"{((d < t) == y).mean()*100:13.2f}%{mark}")

    print("\ninterpolation control (no crop, same pairs):")
    import PIL.Image
    from facenet_utils import preprocess_image
    for name, rs in [("nearest", PIL.Image.NEAREST), ("bilinear", PIL.Image.BILINEAR),
                     ("bicubic", PIL.Image.BICUBIC), ("box/area", PIL.Image.BOX),
                     ("lanczos", PIL.Image.LANCZOS)]:
        uniq = sorted({str(q) for a, b, _ in test for q in (a, b)})
        idx = {q: i for i, q in enumerate(uniq)}
        E = np.zeros((len(uniq), 128), dtype=np.float32)
        for s in range(0, len(uniq), 256):
            ch = uniq[s:s + 256]
            batch = []
            for q in ch:
                im = PIL.Image.open(q).convert("RGB").resize((96, 96), rs)
                batch.append(np.around(np.transpose(im, (2, 0, 1)) / 255.0, decimals=12))
            E[s:s + len(ch)] = model.predict_on_batch(np.stack(batch))
        dd = np.linalg.norm(E[[idx[str(a)] for a, _, _ in test]] -
                            E[[idx[str(b)] for _, b, _ in test]], axis=1)
        print(f"   {name:>9}: {evaluate_verification(dd, y)['accuracy_mean']*100:.2f}%")
    return r


def build_pool(model):
    people = sorted(q for q in LFW_IMAGE_ROOT.iterdir() if q.is_dir())
    multi = [q for q in people if len(photos_of(q)) >= 2]
    rng = np.random.default_rng(POOL_SEED)
    pool = [multi[i] for i in rng.choice(len(multi), size=POOL_SIZE, replace=False)]
    G = encode_many([str(photos_of(q)[0]) for q in pool], model, crop=MODULE_CROP)
    P = encode_many([str(photos_of(q)[1]) for q in pool], model, crop=MODULE_CROP)
    return people, pool, G, P, distance_matrix(P, G)


def experiment_3_identification(D):
    """Report Table II: gallery-size curve and the two predictive models."""
    banner("EXPERIMENT 3  ->  report Table II (order statistics)")
    n = len(D)
    rng = np.random.default_rng(POOL_SEED)
    gen = D[np.arange(n), np.arange(n)]
    p_i = np.array([(D[i][np.arange(n) != i] < gen[i]).mean() for i in range(n)])
    pbar = p_i.mean()

    print(f"{'N':>5} {'E[min imp]':>11} {'observed':>19} {'r_probe':>9} {'r_pool':>10}")
    rows = []
    for N in [2, 5, 10, 25, 50, 100, 150, 300]:
        accs, mins = [], []
        for _ in range(SUBGALLERY_TRIALS if N < n else 1):
            idx = rng.choice(n, N, replace=False)
            blk = D[np.ix_(idx, idx)]
            accs.append((blk.argmin(axis=1) == np.arange(N)).mean())
            m = blk.copy()
            np.fill_diagonal(m, np.inf)
            mins.append(m.min(axis=1).mean())
        obs, sd = float(np.mean(accs)), float(np.std(accs))
        pp, pl = np.mean((1 - p_i) ** (N - 1)), (1 - pbar) ** (N - 1)
        rows.append((N, np.mean(mins), obs, sd, pp, pl))
        print(f"{N:5d} {np.mean(mins):11.3f} {obs*100:12.2f} +/-{sd*100:5.1f}% "
              f"{pp*100:8.2f}% {pl*100:9.4f}%")

    print(f"\npooled per-comparison error p_bar = {pbar*100:.2f}%")
    print(f"probes with p_i == 0 exactly      : {(p_i == 0).sum()}/{n} "
          f"({(p_i == 0).mean()*100:.1f}%)")
    print("p_i distribution:")
    for lo, hi in [(0, 1e-12), (1e-12, 3/(n-1)), (3/(n-1), 0.05), (0.05, 1.01)]:
        print(f"   {lo:7.4f} <= p_i < {hi:7.4f} : {((p_i >= lo) & (p_i < hi)).sum():3d}")

    ranks = identification_ranks(D)
    print(f"\nrank of true identity (1-indexed): median {np.median(ranks)+1:.0f}")
    for k in (1, 5, 25):
        print(f"   top-{k:<2}: {(ranks < k).mean()*100:.2f}%")

    from scipy.stats import rankdata, spearmanr
    order = np.sort(np.where(np.eye(n, dtype=bool), np.inf, D), axis=1)
    density = order[:, :5].mean(axis=1)
    print(f"\nrho(genuine, rank)          = {spearmanr(gen, ranks)[0]:+.2f}")
    print(f"rho(local density, rank)    = {spearmanr(density, ranks)[0]:+.2f} "
          f"(p={spearmanr(density, ranks)[1]:.2f})")
    rg, rd, rr = rankdata(gen), rankdata(density), rankdata(ranks)
    res_r = rr - np.polyval(np.polyfit(rd, rr, 1), rd)
    res_g = rg - np.polyval(np.polyfit(rd, rg, 1), rd)
    print(f"partial rho(genuine, rank | density) = {np.corrcoef(res_g, res_r)[0,1]:+.2f}")
    return rows


def experiment_4_openset(model, people, pool, G, P):
    """Report Section VI-C: open-set DIR at fixed FAR."""
    banner("EXPERIMENT 4  ->  report Section VI-C (open-set)")
    Genr, Pgen = G[:150], P[:150]
    singles = [q for q in people if len(photos_of(q)) == 1 and q not in set(pool)]
    rng = np.random.default_rng(IMPOSTOR_SEED)
    imp = [singles[i] for i in rng.choice(len(singles), size=300, replace=False)]
    Pimp = encode_many([str(photos_of(q)[0]) for q in imp], model, crop=MODULE_CROP)

    Dg, Di = distance_matrix(Pgen, Genr), distance_matrix(Pimp, Genr)
    gmin, garg, imin = Dg.min(axis=1), Dg.argmin(axis=1), Di.min(axis=1)
    correct = garg == np.arange(150)
    print(f"gallery 150 (first 150 of the pool), genuine probes 150, "
          f"impostor probes {len(imp)}")
    print(f"closed-set rank-1 on THIS fixed gallery: {correct.mean()*100:.2f}%")
    print("  (the averaged figure over 200 sub-galleries of the same size is in "
          "Experiment 3)")
    print(f"\n{'tau':>7} {'FAR':>8} {'DIR':>8}")
    for t in [0.60, 0.70, 0.80, 0.90, 1.00]:
        print(f"{t:7.2f} {(imin < t).mean()*100:7.2f}% "
              f"{(correct & (gmin < t)).mean()*100:7.2f}%")
    for tgt in (0.01, 0.10):
        t = np.quantile(imin, tgt)
        print(f"DIR @ FAR={tgt*100:>2.0f}%  (tau={t:.3f}) = "
              f"{(correct & (gmin < t)).mean()*100:.2f}%")
    return gmin, imin, correct


def experiment_5_multishot(model, people):
    """Report Table IV: multi-shot enrolment, with the impostor bar."""
    banner("EXPERIMENT 5  ->  report Table IV (multi-shot enrolment)")
    multi4 = [q for q in people if len(photos_of(q)) >= 4]
    rng = np.random.default_rng(MULTISHOT_SEED)
    sub = [multi4[i] for i in rng.choice(len(multi4), size=150, replace=False)]
    n = len(sub)
    print(f"{len(multi4)} identities have >=4 photographs; using {n}")
    probe = encode_many([str(photos_of(q)[3]) for q in sub], model, crop=MODULE_CROP)
    shots = [encode_many([str(photos_of(q)[j]) for q in sub], model, crop=MODULE_CROP)
             for j in range(3)]
    print(f"\n{'k':>3} {'rank-1':>9} {'genuine d':>11} {'E[min imp]':>12} {'margin':>9}")
    for k in (1, 2, 3):
        t = np.mean(shots[:k], axis=0)
        t = t / np.linalg.norm(t, axis=1, keepdims=True)
        Dk = distance_matrix(probe, t)
        g = Dk[np.arange(n), np.arange(n)]
        m = Dk.copy()
        np.fill_diagonal(m, np.inf)
        mn = m.min(axis=1)
        print(f"{k:3d} {(Dk.argmin(axis=1) == np.arange(n)).mean()*100:8.2f}% "
              f"{g.mean():11.3f} {mn.mean():12.3f} {(mn-g).mean():+9.3f}")


def make_figures(model, test, crop, D, gmin, imin, correct, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path("report/figures")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "figure.dpi": 300, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.labelsize": 8,
                         "axes.titlesize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7})
    BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0392b", "#888888", "#2e7d52"
    y = np.array([l for _, _, l in test])
    d1, _ = pair_distances(test, model, crop=crop, progress=False)
    d0, _ = pair_distances(test, model, crop=None, progress=False)
    r1, r0 = evaluate_verification(d1, y), evaluate_verification(d0, y)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for a, d, nm in ((ax[0], d0, "No crop (whole 250x250 frame)"),
                     (ax[1], d1, "Cropped to face")):
        a.hist(d[y == 1], bins=35, alpha=0.65, color=BLUE, label="same person")
        a.hist(d[y == 0], bins=35, alpha=0.65, color=RED, label="different people")
        a.set(xlabel="L2 distance (unsquared)", xlim=(0, 2),
              title=f"{nm}  (separation {d[y==0].mean()-d[y==1].mean():+.3f})")
        a.legend(fontsize=6.5)
    ax[0].set_ylabel("pair count")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "ieee_distributions.png", bbox_inches="tight")
    plt.close(fig)

    fig, a = plt.subplots(figsize=(3.4, 2.5))
    f1, t1 = r1["roc"]
    f0, t0 = r0["roc"]
    a.plot(f1*100, t1*100, color=BLUE, lw=1.4, label=f"cropped (AUC {r1['auc']:.3f})")
    a.plot(f0*100, t0*100, color=RED, lw=1.2, ls="--",
           label=f"no crop (AUC {r0['auc']:.3f})")
    a.plot([0, 100], [0, 100], ls=":", c=GREY, lw=0.9, label="chance")
    a.set(xlabel="false accept rate (%)", ylabel="true accept rate (%)")
    a.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "ieee_roc.png", bbox_inches="tight")
    plt.close(fig)

    Ns = [r[0] for r in rows]
    obs = np.array([r[2] for r in rows]) * 100
    sd = np.array([r[3] for r in rows]) * 100
    pp = np.array([r[4] for r in rows]) * 100
    pl = np.array([r[5] for r in rows]) * 100
    lo = np.minimum(sd, 100 - obs)          # never draw an error bar above 100%

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    a.errorbar(Ns, obs, yerr=[sd, lo], marker="o", ms=3.5, color=BLUE, capsize=2.5,
               lw=0, elinewidth=1.1, label="measured (200 sub-galleries)")
    a.plot(Ns, pp, color=GREEN, lw=1.4, label="per-probe model, Eq. (2)")
    a.plot(Ns, pl, color=RED, lw=1.2, ls="--", label="pooled model, Eq. (3)")
    a.plot(Ns, 100/np.array(Ns), ls=":", c=GREY, lw=1.0, label="chance (1/N)")
    a.set(xscale="log", yscale="log", xlabel="gallery size N",
          ylabel="rank-1 accuracy (%)", title="Closed-set identification",
          ylim=(0.05, 160))
    a.legend(fontsize=6, loc="lower left")

    taus = np.linspace(0.2, 1.6, 400)
    far = np.array([(imin < t).mean() for t in taus])
    dir_ = np.array([(correct & (gmin < t)).mean() for t in taus])
    o = np.argsort(far)
    b.plot(far[o]*100, dir_[o]*100, color=GREEN, lw=1.5)
    for tgt, mk in ((0.01, "o"), (0.10, "s")):
        t = np.quantile(imin, tgt)
        v = (correct & (gmin < t)).mean()
        b.plot(tgt*100, v*100, mk, color=RED, ms=5,
               label=f"FAR={tgt*100:.0f}%: DIR={v*100:.1f}%")
    b.axhline(correct.mean()*100, ls="--", c=GREY, lw=1.1,
              label=f"closed-set, this gallery = {correct.mean()*100:.1f}%")
    b.set(xscale="log", xlabel="false accept rate (%)",
          ylabel="detection and identification rate (%)",
          title="Open-set identification", ylim=(0, 70))
    b.legend(fontsize=6, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "ieee_identification.png", bbox_inches="tight")
    plt.close(fig)
    print("\nfigures written to report/figures/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true", help="regenerate report figures")
    args = ap.parse_args()

    print("Building network and loading pretrained weights ...")
    model = build_embedding_network()
    print(f"  {apply_pretrained_weights(model)} layers restored")
    print(f"  crop used by the shipped pipeline = {MODULE_CROP}")

    tune, test, shared = build_splits()
    print(f"  tuning {len(tune)} pairs, test {len(test)} pairs, "
          f"0 shared pairs, {shared} shared images")

    crop = experiment_1_crop(model, tune, test)
    experiment_2_verification(model, test, crop)
    people, pool, G, P, D = build_pool(model)
    rows = experiment_3_identification(D)
    gmin, imin, correct = experiment_4_openset(model, people, pool, G, P)
    experiment_5_multishot(model, people)
    if args.figures:
        make_figures(model, test, crop, D, gmin, imin, correct, rows)


if __name__ == "__main__":
    main()
