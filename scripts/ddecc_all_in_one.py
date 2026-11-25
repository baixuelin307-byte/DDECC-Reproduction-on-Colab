# ===== DDECC reproduction: check + results + figures + comparison with the paper + GitHub package =====
# One cell, no GPU. Reads DDECC_results (never modifies it); writes DDECC_results/figures_v3/,
# DDECC_reproduction_github/ and DDECC_reproduction_github.zip on Drive.



# ============================== 1. setup ==============================
import os, sys, re, json, math, time, subprocess
from pathlib import Path
from google.colab import drive
if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")

# ============================== settings ==============================
ROOT  = Path("/content/drive/MyDrive/DDECC_results")    # your existing results folder
REPO  = Path("/content/DDECC")
CODES = [("BCH", 63, 36), ("BCH", 63, 45), ("BCH", 63, 51),
         ("POLAR", 64, 32), ("POLAR", 64, 48),
         ("POLAR", 128, 64), ("POLAR", 128, 86), ("POLAR", 128, 96),
         ("LDPC", 49, 24), ("LDPC", 121, 60), ("LDPC", 121, 70), ("LDPC", 121, 80),
         ("MACKAY", 96, 48), ("CCSDS", 128, 64)]
# training (same as your earlier runs)
N_DEC, D_MODEL, HEADS = 6, 128, 8
EPOCHS, BATCH_SIZE, LR, SIGMA = 2000, 128, 1e-4, 0.01
WORKERS, SAVE_EVERY = min(4, os.cpu_count() or 1), 50
RETRAIN = {}            # e.g. {("BCH", 63, 36): 1, ("POLAR", 128, 86): 1}  -> code: seed
# testing
EBNO = [4, 5, 6]
TEST_MODES = ["reg", "LS"]
MIN_FER, MIN_FRAMES, MAX_FRAMES = 500, 1e5, 1e7
STRICT_500 = False      # True: points that stopped with < 500 frame errors (old tests) are tested again in step 5
TEST_BATCH, SEED, SAVE_EVERY_S = 2048, 42, 120
# ======================================================================

RUNS, TESTS, FIGS = ROOT / "runs_v3", ROOT / "eval_v3", ROOT / "figures_v3"
for p in (RUNS, TESTS, FIGS):
    p.mkdir(parents=True, exist_ok=True)
MODE_NAME = {"reg": "Regular Reverse Diffusion", "LS": "Line Search Reverse Diffusion"}


def read(p):
    try:
        return Path(p).read_text(errors="ignore")
    except OSError:
        return ""


def read_int(p):
    try:
        return int(Path(p).read_text().strip())
    except (OSError, ValueError):
        return None


def nums(s):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace("np.float64", ""))]


def max_epoch(d):
    eps = [int(x) for x in re.findall(r"Training epoch (\d+)", read(Path(d) / "logging.txt"))]
    return max(eps) if eps else 0


def name(c):
    return f"{c[0]}({c[1]},{c[2]})"


def folder_time(p):
    # run folders end with __DD_MM_YYYY_HH_MM_SS: order by that time, not by Drive's file dates
    m = re.search(r"__(\d\d)_(\d\d)_(\d{4})_(\d\d)_(\d\d)_(\d\d)$", Path(p).name)
    if m:
        d, mo, y, h, mi, s = map(int, m.groups())
        return (y, mo, d, h, mi, s)
    return (0,) * 6


def tag(c):
    return f"{c[0]}_{c[1]}_{c[2]}"


def run_dir(c, seed):
    return RUNS / f"{c[0]}_n{c[1]}_k{c[2]}_N{N_DEC}_d{D_MODEL}_seed{seed}"


def save_json(path, obj):
    for _ in range(10):                      # Drive can fail for a moment: retry
        try:
            tmp = Path(str(path) + ".tmp")
            tmp.write_text(json.dumps(obj, indent=1))
            os.replace(tmp, path)
            return
        except OSError as e:
            print("save failed, retrying:", e, flush=True)
            time.sleep(10)


# ---------------------------- which model ----------------------------
def find_model(c):
    # -> dict(state=trained|resume|fresh, model=..., src=new|old, dir=..., seed=..)
    t, n, k = c
    seed = RETRAIN.get(c, 42)
    d = run_dir(c, seed)
    e = dict(dir=str(d), seed=seed)
    if (d / "train_done.txt").exists() and (d / "best_model").exists():
        return {**e, "state": "trained", "model": str(d / "best_model"), "src": "new"}
    old = sorted(ROOT.glob(f"{t}__Code_n_{n}_k_{k}__*"), key=folder_time)
    if c not in RETRAIN:
        done = [x for x in old if (x / "best_model").exists() and max_epoch(x) >= EPOCHS]
        if done:
            return {**e, "state": "trained", "model": str(done[-1] / "best_model"), "src": "old"}
    cands = []
    for x in [d] + ([] if c in RETRAIN else old):
        ep = read_int(x / "last_epoch.txt")
        if ep and (x / "last_model").exists():
            cands.append((ep, str(x / "last_model")))
    if cands:
        ep, path = max(cands)
        return {**e, "state": "resume", "resume": path, "start": ep + 1}
    return {**e, "state": "fresh"}


# ---------------------------- results ----------------------------
PROG = re.compile(r"\[(reg|LS)\] EbN0=(\d+): ([\d.eE+]+) frames, (\d+) frame errors, (\d+) bit errors")


def res_official(c):
    t, n, k = c
    out = {}
    for line in read(ROOT / "eval_official" / f"log_{tag(c)}.txt").splitlines():
        m = PROG.search(line)
        if not m or int(m[2]) not in EBNO:
            continue
        mode, e, fr, fe, be = m[1], int(m[2]), float(m[3]), int(m[4]), int(m[5])
        done = line.rstrip().endswith("done")
        old = out.get((mode, e))
        if old and old["done"] and not done:
            continue
        out[(mode, e)] = dict(ber=be / (fr * n), fer=fe / fr, frames=fr, fe=fe, done=done,
                              src=f"eval_official/log_{tag(c)}.txt")
    return out


def res_eval_resume(c, model):
    t, n, k = c
    out = {}
    for f, mode in [(f"eval_state_{tag(c)}.json", "reg"), (f"eval_state_{tag(c)}_reg.json", "reg"),
                    (f"eval_state_{tag(c)}_LS.json", "LS")]:
        s = json.loads(read(ROOT / "eval_resume" / f) or "{}")
        if s.get("meta", {}).get("model_path") != model:
            continue
        for e, st in s.items():
            if e != "meta" and int(e) in EBNO and st.get("bit_err"):
                fr = st["frames"]
                out.setdefault((mode, int(e)), dict(ber=st["bit_err"][-1] / (fr * n), fer=st["frame_err"][-1] / fr,
                                                    frames=fr, fe=st["frame_err"][-1], done=bool(st.get("done")),
                                                    src=f"eval_resume/{f}"))
    return out


def _main_section(txt, section):
    i = txt.rfind(section)
    if i < 0:
        return {}
    seg = txt[i + len(section):]
    for other in MODE_NAME.values():
        j = seg.find(other)
        if j >= 0:
            seg = seg[:j]
    m = re.search(r"# of testing samples: \[([^\]]*)\]", seg)
    frames = nums(m.group(1)) if m else []
    out = {}
    for idx, e in enumerate([4, 5, 6]):          # Main.py tests range(4, 7)
        mb = re.search(rf"Test EbN0={e}, BER=\[([^\]]*)\]", seg)
        mf = re.search(rf"Test EbN0={e}, FER_DDPM=\[([^\]]*)\]", seg)
        if mb and mf and e in EBNO:
            fr = frames[idx] if idx < len(frames) else None
            fer = nums(mf.group(1))[-1]
            out[e] = dict(ber=nums(mb.group(1))[-1], fer=fer, frames=fr, fe=round(fer * fr) if fr else None, done=True)
    return out


def res_main_logs(c, model):
    t, n, k = c
    mdir = Path(model).parent
    texts = [(read(mdir / "logging.txt"), mdir.name + "/logging.txt")]
    for d in sorted(ROOT.glob(f"{t}__Code_n_{n}_k_{k}__*"), key=folder_time, reverse=True):
        if d != mdir and f"resume='{model}'" in read(d / "logging.txt"):
            texts.append((read(d / "logging.txt"), d.name + "/logging.txt"))   # old "test only" runs
    for lf in ROOT.glob(f"{t}_n{n}_k{k}_N*_d*.log"):
        for part in reversed(re.split(r"(?m)^(?=Path to model/logs:)", read(lf))):
            first = part.split("\n", 1)[0]
            if Path(first.split(":", 1)[-1].strip()).name == mdir.name or f"resume='{model}'" in part:
                texts.append((part, lf.name))
    out = {}
    for txt, src in texts:
        for mode, sec in MODE_NAME.items():
            for e, v in _main_section(txt, sec).items():
                v = {**v, "src": src}
                if (mode, e) not in out or rank(v) > rank(out[(mode, e)]):
                    out[(mode, e)] = v
    return out


def v3_file(model, mode):
    return TESTS / f"{Path(model).parent.name}_{mode}.json"


def res_v3(c, model):
    t, n, k = c
    out = {}
    for mode in MODE_NAME:
        s = json.loads(read(v3_file(model, mode)) or "{}")
        if s.get("meta", {}).get("model_path") != model:
            continue
        for e, st in s.items():
            if e != "meta" and int(e) in EBNO and st.get("bit_err"):
                fr = st["frames"]
                out[(mode, int(e))] = dict(ber=st["bit_err"][-1] / (fr * n), fer=st["frame_err"][-1] / fr,
                                           frames=fr, fe=st["frame_err"][-1], done=bool(st.get("done")),
                                           src=f"eval_v3/{v3_file(model, mode).name}")
    return out


def results(c, m):
    # every (mode, Eb/N0) -> result of THIS model; a finished one beats an unfinished one
    if m["state"] != "trained":
        return {}
    model = m["model"]
    sources = [res_v3(c, model)]
    if m["src"] == "old":                    # the old result files belong to the old models
        sources += [res_official(c), res_eval_resume(c, model), res_main_logs(c, model)]
    out = {}
    for src in sources:
        for key, v in src.items():
            if key not in out or rank(v) > rank(out[key]):
                out[key] = v
    return out


def protocol_ok(r):
    # DDECC paper: >= 500 frame errors (and >= 1e5 frames); the 1e7-frame cap ends a point too.
    # None = unknown (an old Main.py log without the frame count)
    if not r["done"]:
        return False
    if r["fe"] is None and r["frames"] is None:
        return None
    return (r["fe"] or 0) >= MIN_FER or (r["frames"] or 0) >= MAX_FRAMES


def rank(r):
    # when one point was tested several times: finished > unfinished, then the protocol-compliant
    # one (known > unknown > below 500 errors), then the one with more frame errors
    ok = protocol_ok(r)
    return (r["done"], 2 if ok else 1 if ok is None else 0, r["fe"] or 0, r["frames"] or 0)


def is_done(r):
    return bool(r) and r["done"] and (protocol_ok(r) is not False if STRICT_500 else True)


def mark(r):
    # ✓ finished; ✓? finished, frame count unknown; ✓¹ finished with < 500 errors; … unfinished; — none
    if not r:
        return "—"
    if not r["done"]:
        return "…"
    ok = protocol_ok(r)
    return "✓" if ok else "✓?" if ok is None else "✓¹"


def check(verbose=True):
    S = {}
    for c in CODES:
        m = find_model(c)
        S[c] = dict(model=m, res=results(c, m))
    if verbose:
        head = "".join(f"{m}{e}".rjust(7) for m in TEST_MODES for e in EBNO)
        print(f"{'code':<15}{'training':<24}{head}   model")
        print("-" * 118)
        for c, s in S.items():
            m = s["model"]
            st = {"trained": "✅ trained" + (" (old)" if m.get("src") == "old" else " (new)"),
                  "resume": f"⏸ resume {m.get('start')}→{EPOCHS}", "fresh": "⬜ not trained"}[m["state"]]
            cells = ""
            for mode in TEST_MODES:
                for e in EBNO:
                    r = s["res"].get((mode, e))
                    cells += mark(r).rjust(7)
            where = (m.get("model") or m.get("resume") or m["dir"]).replace(str(ROOT) + "/", "")
            print(f"{name(c):<15}{st:<24}{cells}   {where}")
        n_tr = sum(s["model"]["state"] == "trained" for s in S.values())
        n_te = sum(all(is_done(s["res"].get((mo, e))) for mo in TEST_MODES for e in EBNO) for s in S.values())
        print("-" * 118)
        print(f"trained {n_tr}/{len(S)}, fully tested {n_te}/{len(S)}   (✓ finished, … unfinished, — no result)")
        print(f"✓¹ finished with < {MIN_FER} frame errors before {MAX_FRAMES:.0e} frames (an old test that stopped at Main.py's "
              f"100 errors): valid but less precise; STRICT_500 = True re-tests these in step 5")
        print("✓? finished, but the old Main.py log does not record the frame count")
    return S


def lnb(r):
    return -math.log(r["ber"]) if r["ber"] > 0 else float("inf")


# ---------------------------- DDECC repo + patches ----------------------------
def rep(src, old, new, what):
    k = src.count(old)
    assert k == 1, f"patch '{what}' failed: the original text occurs {k} times (expected 1)"
    return src.replace(old, new)


def ensure_repo():
    # clone the DDECC repo; restore Main.py/DDECC.py and patch them again (safe to repeat)
    if not (REPO / "Main.py").exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/yoniLc/DDECC.git", str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--", "Main.py", "DDECC.py"], check=True)
    src = (REPO / "Main.py").read_text()
    src = rep(src, "def main(args):", '''def _safe_save(obj, path, retries=10, wait=30):
    for i in range(retries):            # temp file + rename; retry on short Drive outages
        try:
            torch.save(obj, path + '.tmp')
            os.replace(path + '.tmp', path)
            return True
        except Exception as e:
            print(f'[save] {path} failed ({e}); retry {i + 1}/{retries}', flush=True)
            time.sleep(wait)
    return False


def _safe_write(path, text, retries=10, wait=30):
    for i in range(retries):
        try:
            with open(path + '.tmp', 'w') as f:
                f.write(text)
            os.replace(path + '.tmp', path)
            return True
        except Exception as e:
            print(f'[write] {path} failed ({e}); retry {i + 1}/{retries}', flush=True)
            time.sleep(wait)
    return False


def main(args):''', "helpers")
    src = rep(src, "    parser.add_argument('--sigma', type=float, default=0.01)\n",
              "    parser.add_argument('--sigma', type=float, default=0.01)\n"
              "    parser.add_argument('--out_dir', type=str, default='')\n"
              "    parser.add_argument('--resume', type=str, default='')\n"
              "    parser.add_argument('--start_epoch', type=int, default=1)\n"
              "    parser.add_argument('--save_every', type=int, default=50)\n"
              "    parser.add_argument('--no_test', action='store_true')\n", "arguments")
    src = rep(src, "    os.makedirs(model_dir, exist_ok=True)\n",
              "    if args.out_dir:\n        model_dir = args.out_dir\n    os.makedirs(model_dir, exist_ok=True)\n", "out_dir")
    src = rep(src, "    model = DDECCT(args, device=device,dropout=0).to(device)\n    model.ema.register(model)\n",
              "    if args.resume:\n"
              "        model = torch.load(args.resume, map_location=device, weights_only=False).to(device)\n"
              "        model.device = device\n"
              "        logging.info(f'Resumed from {args.resume}, start_epoch={args.start_epoch}')\n"
              "    else:\n"
              "        model = DDECCT(args, device=device,dropout=0).to(device)\n"
              "        model.ema.register(model)\n", "model creation")
    src = rep(src, "    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)\n",
              "    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)\n"
              "    for _ in range(args.start_epoch - 1):\n"
              "        scheduler.step()\n"
              "    _opt = os.path.join(os.path.dirname(args.resume), 'last_optim.pt') if args.resume else ''\n"
              "    if _opt and os.path.exists(_opt):\n"
              "        _cur = optimizer.state_dict()\n"
              "        _cur['state'] = torch.load(_opt, map_location=device, weights_only=False)['state']\n"
              "        optimizer.load_state_dict(_cur)\n"
              "        logging.info('Adam state restored')\n", "scheduler")
    src = rep(src, "    best_loss = float('inf')\n    for epoch in range(1, args.epochs + 1):\n",
              "    best_loss = float('inf')\n"
              "    _bl = os.path.join(os.path.dirname(args.resume), 'best_loss.txt') if args.resume else ''\n"
              "    if _bl and os.path.exists(_bl):\n"
              "        best_loss = float(open(_bl).read())\n"
              "    for epoch in range(args.start_epoch, args.epochs + 1):\n", "epoch loop")
    src = rep(src, "            torch.save(model, os.path.join(args.path, 'best_model'))\n            logging.info(f'Model Saved')\n",
              "            if _safe_save(model, os.path.join(args.path, 'best_model')):\n"
              "                _safe_write(os.path.join(args.path, 'best_loss.txt'), repr(best_loss))\n"
              "            logging.info(f'Model Saved')\n"
              "        if epoch % args.save_every == 0 or epoch == args.epochs:\n"
              "            if (_safe_save(model, os.path.join(args.path, 'last_model')) and\n"
              "                    _safe_save(optimizer.state_dict(), os.path.join(args.path, 'last_optim.pt'))):\n"
              "                _safe_write(os.path.join(args.path, 'last_epoch.txt'), str(epoch))\n", "save")
    src = rep(src, "        if epoch % (args.epochs//2) == 0 or epoch in [1,25]:\n",
              "        if (not args.no_test) and (epoch % (args.epochs//2) == 0 or epoch in [1,25]):\n", "mid test")
    src = rep(src, "    logging.info('Loading Best Model')\n    model = torch.load(os.path.join(args.path, 'best_model')).to(device)\n",
              "    _safe_write(os.path.join(args.path, 'train_done.txt'), str(args.epochs))\n"
              "    if args.no_test:\n"
              "        logging.info('Training finished (testing is done by the notebook)')\n"
              "        return\n"
              "    logging.info('Loading Best Model')\n"
              "    model = torch.load(os.path.join(args.path, 'best_model'), weights_only=False).to(device)\n", "final test")
    compile(src, "Main.py", "exec")
    (REPO / "Main.py").write_text(src)
    dsrc = (REPO / "DDECC.py").read_text()
    dsrc = rep(dsrc, "        #Single sampling from the real p dist.\n",
               "        #Single sampling from the real p dist.\n        yt = yt.cpu()  # [patch] reverse step on the CPU\n", "yt cpu")
    dsrc = rep(dsrc, "        t = sum_syndrome\n", "        t = sum_syndrome.cpu()  # [patch] CPU index for the CPU betas\n", "t cpu")
    compile(dsrc, "DDECC.py", "exec")
    (REPO / "DDECC.py").write_text(dsrc)


SETUP_OK = True
print("Setup done. Results folder:", ROOT, "(found)" if ROOT.exists() else "(NOT FOUND)")

# ============================== 2. check ==============================

S = check()

# ============================== 3. results + figures ==============================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

S = check(verbose=False)
R = {c: s["res"] for c, s in S.items()}

for mode in TEST_MODES:
    print("\n" + "=" * 62 + f"\n−ln(BER), {MODE_NAME[mode]}   (* unfinished, ¹ < {MIN_FER} frame errors, ? frame count unknown, — no result)")
    print(f"{'code':<16}" + "".join(f"{e} dB".rjust(10) for e in EBNO))
    for c in CODES:
        row = ""
        for e in EBNO:
            r = R[c].get((mode, e))
            tail = {"—": "", "…": " *", "✓": "  ", "✓?": "? ", "✓¹": "¹ "}[mark(r)]
            row += "—".rjust(10) if not r else f"{lnb(r):8.2f}{tail}"
        print(f"{name(c):<16}{row}")

print("\nsources:")
for c in CODES:
    print(f"  {name(c):<15}", "; ".join(sorted({r['src'] for r in R[c].values()})) or "no results")

rows = ["code,n,k,decoding,ebno,neg_ln_ber,ber,fer,frames,frame_errors,finished,model,source"]
for c in CODES:
    for (mode, e), r in sorted(R[c].items()):
        rows.append(f"{c[0]},{c[1]},{c[2]},{mode},{e},{lnb(r):.4f},{r['ber']:.6e},{r['fer']:.6e},{r['frames'] or ''},"
                    f"{'' if r['fe'] is None else r['fe']},{r['done']},{S[c]['model'].get('model', '')},{r['src']}")
(FIGS / "results.csv").write_text("\n".join(rows) + "\n")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # fixed order, one per code
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
mpl.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False})
FAMILIES = ["BCH", "POLAR", "LDPC", "MACKAY", "CCSDS"]

for mode in TEST_MODES:
    for ycol, ylabel, logy in [("lnb", "−ln(BER)  (higher is better)", False), ("ber", "BER", True)]:
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.9), constrained_layout=True)
        for ax, fam in zip(axes, FAMILIES):
            for j, c in enumerate([c for c in CODES if c[0] == fam]):
                pts = [(e, R[c][(mode, e)]) for e in EBNO if (mode, e) in R[c] and R[c][(mode, e)]["ber"] > 0]
                if pts:
                    ax.plot([e for e, _ in pts], [lnb(r) if ycol == "lnb" else r["ber"] for _, r in pts],
                            color=COLORS[j], marker=MARKERS[j], lw=2, ms=7, markeredgecolor="white",
                            markeredgewidth=1.2, label=f"({c[1]},{c[2]})")
            if ax.lines:
                ax.legend(fontsize=9)
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=INK2)
            ax.set_title(fam, fontweight="bold", loc="left")
            ax.set_xlim(min(EBNO) - 0.2, max(EBNO) + 0.2); ax.set_xticks(EBNO); ax.set_xlabel("Eb/N0 (dB)")
            if logy:
                ax.set_yscale("log")
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"DDECC reproduction (N={N_DEC}, d={D_MODEL}) — {ylabel.split()[0]} vs Eb/N0, {MODE_NAME[mode]}",
                     x=0.01, ha="left", fontsize=12)
        fig.savefig(FIGS / f"{ycol}_{mode}.png", dpi=300, bbox_inches="tight")
        plt.show()

cols = [(m, e) for m in TEST_MODES for e in EBNO]
vals = np.array([[lnb(R[c][key]) if key in R[c] else np.nan for key in cols] for c in CODES])
fin = vals[np.isfinite(vals)]
fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2.6, 0.42 * len(CODES) + 1.4), constrained_layout=True)
cmap = mpl.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
ax.imshow(np.where(np.isfinite(vals), vals, np.nan), cmap=cmap, aspect="auto",
          vmin=fin.min() if fin.size else 0, vmax=fin.max() if fin.size else 1)
for i, c in enumerate(CODES):
    for j, key in enumerate(cols):
        v, r = vals[i, j], R[c].get(key)
        txt = "—" if np.isnan(v) else ("inf" if np.isinf(v) else f"{v:.2f}") + ("" if r is None or r["done"] else "*")
        dark = np.isfinite(v) and fin.size and v > fin.min() + 0.55 * (fin.max() - fin.min())
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if dark else INK)
ax.set_xticks(range(len(cols)), [f"{m}\n{e} dB" for m, e in cols])
ax.set_yticks(range(len(CODES)), [name(c) for c in CODES])
for x in range(len(EBNO), len(cols), len(EBNO)):
    ax.axvline(x - 0.5, color="white", lw=3)
ax.tick_params(length=0); ax.grid(False); ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("−ln(BER)   (reg = regular, LS = line search; * = unfinished)", loc="left", fontsize=10, pad=34)
fig.savefig(FIGS / "table_neglnBER.png", dpi=300, bbox_inches="tight")
plt.show()
print("saved to", FIGS, ":", ", ".join(sorted(p.name for p in FIGS.iterdir())))

# ============================== 4. comparison with the DDECC paper ==============================

import matplotlib.pyplot as plt
import matplotlib as mpl

# DDECC paper (ICLR 2023), N=6, d=128 — Table 1 (line search) and Appendix E Table 4 (Δ = LS − regular)
PAPER_LS = {("BCH",63,36):[5.11,7.09,9.82], ("BCH",63,45):[5.61,7.94,11.36], ("BCH",63,51):[5.26,7.40,10.49],
            ("POLAR",64,32):[6.93,9.51,12.79], ("POLAR",64,48):[5.96,8.04,10.98], ("POLAR",128,64):[9.11,12.90,16.30],
            ("POLAR",128,86):[7.60,10.81,15.17], ("POLAR",128,96):[7.16,10.30,13.19], ("LDPC",49,24):[5.88,8.27,11.42],
            ("LDPC",121,60):[5.38,8.73,14.17], ("LDPC",121,70):[6.79,11.13,16.93], ("LDPC",121,80):[7.59,12.17,16.89],
            ("MACKAY",96,48):[8.12,11.88,15.93], ("CCSDS",128,64):[7.81,12.48,17.66]}
PAPER_DELTA = {("BCH",63,36):[-0.11,-0.02,0.04], ("BCH",63,45):[-0.13,-0.06,0.73], ("BCH",63,51):[0.02,0.34,1.04],
               ("POLAR",64,32):[-0.11,0.07,0.65], ("POLAR",64,48):[-0.01,0.07,0.71], ("POLAR",128,64):[-0.12,-0.20,1.00],
               ("POLAR",128,86):[-0.32,-0.21,-0.40], ("POLAR",128,96):[-0.15,0.00,-0.13], ("LDPC",49,24):[-0.12,-0.13,-0.41],
               ("LDPC",121,60):[-0.18,-0.47,-0.23], ("LDPC",121,70):[-0.18,0.00,-0.11], ("LDPC",121,80):[-0.28,-0.21,-0.27],
               ("MACKAY",96,48):[-0.19,-0.33,-0.13], ("CCSDS",128,64):[-0.26,-0.11,0.37]}
PAPER = {"LS": PAPER_LS, "reg": {c: [round(l - d, 2) for l, d in zip(PAPER_LS[c], PAPER_DELTA[c])] for c in PAPER_LS}}
PE = [4, 5, 6]

S = check(verbose=False)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
mpl.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "legend.frameon": False})
for mode in TEST_MODES:
    print("\n" + "=" * 78 + f"\n{MODE_NAME[mode]}: reproduction − paper (N=6, d=128), −ln(BER)")
    print(f"{'code':<16}" + "".join(f"{e} dB".rjust(22) for e in PE))
    fig, axes = plt.subplots(2, 7, figsize=(20, 6.2), constrained_layout=True)
    for ax, c in zip(axes.flat, CODES):
        paper = PAPER[mode].get(c)
        ours = [S[c]["res"].get((mode, e)) for e in PE]
        cells = ""
        for e, p, r in zip(PE, paper or [None] * 3, ours):
            cells += ("—" if r is None or p is None else f"{lnb(r):.2f} vs {p:.2f} ({lnb(r) - p:+.2f})").rjust(22)
        print(f"{name(c):<16}{cells}")
        if paper:
            ax.plot(PE, paper, color="#eb6834", ls="--", marker="s", lw=2, ms=6, label=f"DDECC paper ({mode}, N=6, d=128)")
        pts = [(e, lnb(r)) for e, r in zip(PE, ours) if r and r["ber"] > 0]
        if pts:
            ax.plot([e for e, _ in pts], [v for _, v in pts], color="#2a78d6", marker="o", lw=2, ms=6,
                    label=f"reproduction ({mode})")
        if paper and len(pts) == 3:
            ax.text(0.04, 0.93, "Δ " + " / ".join(f"{v - p:+.2f}" for (_, v), p in zip(pts, paper)),
                    transform=ax.transAxes, color=INK2, fontsize=8, va="top")
        ax.set_title(name(c), fontsize=10, loc="left", color=INK, fontweight="bold")
        ax.set_xticks(PE); ax.set_xlim(3.8, 6.2)
    for ax in axes[1]:
        ax.set_xlabel("Eb/N0 (dB)")
    for ax in axes[:, 0]:
        ax.set_ylabel("−ln(BER)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=10)
    src = "Table 1" if mode == "LS" else "Table 1 − Δ (Appendix E, Table 4)"
    fig.suptitle(f"{MODE_NAME[mode]}: DDECC reproduction vs. DDECC paper ({src}); Δ = reproduction − paper at 4/5/6 dB",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(FIGS / f"vs_paper_{mode}.png", dpi=300, bbox_inches="tight")
    plt.show()
print("\nsaved to", FIGS)

# ============================== 5. package for GitHub ==============================
# ===== Collect the DDECC reproduction results for GitHub (logs, test results, figures; no models) =====
# read-only on DDECC_results; writes one new folder + one zip on Drive. No GPU needed, runs on its own.
import os, re, shutil
from pathlib import Path
from google.colab import drive
if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")

DRIVE = Path("/content/drive/MyDrive")
ROOT = DRIVE / "DDECC_results"
OUT = DRIVE / "DDECC_reproduction_github"          # the folder to push
SKIP = {"best_model", "last_model", "last_optim.pt"}  # model files: too large for GitHub
MAX_MB = 50                                         # GitHub refuses files > 100 MB; skip anything > 50 MB

if OUT.exists():
    shutil.rmtree(OUT)
copied, skipped = [], []


def put(src, rel):
    src = Path(src)
    if src.name in SKIP or src.suffix in (".tmp", ".pt", ".pth", ".bak") or ".bak" in src.name:
        return
    mb = src.stat().st_size / 1e6
    if mb > MAX_MB:
        skipped.append(f"{src.relative_to(DRIVE)} ({mb:.0f} MB)")
        return
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel)


# 1) training runs: logging.txt and small text files of every run folder (old + runs_v3)
for d in sorted(ROOT.glob("*__Code_n_*")) + sorted((ROOT / "runs_v3").glob("*")):
    if d.is_dir():
        for f in d.iterdir():
            if f.is_file() and f.suffix in ("", ".txt") and f.name not in SKIP:
                put(f, f"training_logs/{d.name}/{f.name}")
# 2) the old runner's console logs and .done flags
for f in sorted(ROOT.glob("*.log")) + sorted(ROOT.glob("*.done")):
    put(f, f"runner_logs/{f.name}")
# 3) test results
for sub in ("eval_official", "eval_resume", "eval_v3"):
    for f in sorted((ROOT / sub).glob("*")):
        if f.is_file():
            put(f, f"test_results/{sub}/{f.name}")
# 4) tables + figures made by the notebook (steps 6 and 7)
for sub in ("figures_v3", "figures"):
    for f in sorted((ROOT / sub).glob("*")):
        if f.is_file():
            put(f, f"figures/{f.name}" if sub == "figures_v3" else f"figures/old/{f.name}")
# 5) the notebook itself, if it is in Colab Notebooks
for f in (DRIVE / "Colab Notebooks").glob("DDECC_reproduction*.ipynb"):
    put(f, f.name)

# 6) README with the result table (from step 6's results.csv, if it exists)
csv = OUT / "figures" / "results.csv"
table = ""
if csv.exists():
    rows = [l.split(",") for l in csv.read_text().splitlines()[1:]]
    for mode, title in [("reg", "Regular reverse diffusion"), ("LS", "Line-search reverse diffusion")]:
        table += f"\n### {title}: −ln(BER)\n\n| Code | 4 dB | 5 dB | 6 dB |\n|---|---|---|---|\n"
        codes = list(dict.fromkeys((r[0], r[1], r[2]) for r in rows))
        for t, n, k in codes:
            v = {r[4]: float(r[5]) for r in rows if (r[0], r[1], r[2]) == (t, n, k) and r[3] == mode}
            table += f"| {t}({n},{k}) | " + " | ".join(f"{v[e]:.2f}" if e in v else "—" for e in ("4", "5", "6")) + " |\n"
(OUT / "README.md").write_text(f'''# DDECC reproduction

Reproduction of *Denoising Diffusion Error Correction Codes* (Choukroun & Wolf, ICLR 2023)
with the code of https://github.com/yoniLc/DDECC, for 14 codes at Eb/N0 = 4, 5, 6 dB.

- Model: N = 6, d = 128, h = 8; training: 2000 epochs × 1000 batches, batch 128, lr 1e-4 (cosine to 5e-6), β = 0.01
- Test: per Eb/N0, until > 500 frame errors and > 1e5 frames, or 1e7 frames;
  regular and line-search reverse diffusion; metric −ln(BER), higher is better
{table}
## Contents

| folder | content |
|---|---|
| `training_logs/` | `logging.txt` (and small status files) of every training run |
| `runner_logs/` | console output of the training runner |
| `test_results/` | resumable test logs and states (`eval_official`, `eval_resume`, `eval_v3`) |
| `figures/` | `results.csv`, BER / −ln(BER) figures, comparison with the paper |

Trained models (`best_model`) are not included (size).
''')

shutil.make_archive(str(DRIVE / "DDECC_reproduction_github"), "zip", OUT)
size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
print(f"{len(copied)} files, {size:.1f} MB  ->  {OUT}")
print(f"zip: {DRIVE / 'DDECC_reproduction_github.zip'}")
if skipped:
    print("skipped (too large for GitHub):", *skipped, sep="\n  ")
if not csv.exists():
    print("note: no figures_v3/results.csv yet -> run steps 1, 6, 7 of the notebook first for the README table")
