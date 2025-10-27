# task2_cancer_patterns_final.py
# SC4020 Project – Task 2: Mining Cancer Feature Patterns
# 
# finds simple repeating patterns in the Breast Cancer dataset.
# bins all features into low/med/high, builds small ordered lists
# per patient, and runs a fast gsp search to see what combos show up alot.

import pandas as pd
import numpy as np
import itertools
import os
import time
from datetime import datetime
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

# ------------- basic config -------------
CSV_PATH = "breast-cancer.csv"
TARGET_COL = "diagnosis"
ID_COL = "id"
N_BINS = 3
BIN_LABELS = {0: "low", 1: "med", 2: "high"}
MIN_SUPPORT = 0.25      # higher = faster
MAX_LEN = 2             # shorter pattern length = faster
TOP_K = 3               # shorter patient seq = faster
RESULTS_DIR = "task2_results"

# ------------- helper funcs -------------
def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def encode_target(s):
    # M -> 1 (malig), B -> 0 (benign)
    return s.map({"M": 1, "B": 0})

def zscore(df):
    sc = StandardScaler()
    arr = sc.fit_transform(df)
    return pd.DataFrame(arr, columns=df.columns, index=df.index)

def discretize(df, n_bins=3, strategy="quantile"):
    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    arr = kb.fit_transform(df)
    arr = np.nan_to_num(arr, nan=0.0)
    return pd.DataFrame(arr.astype(int), columns=df.columns, index=df.index)

# ------------- build short sequences -------------
def build_sequences(X_disc, X_z, top_k=3, bin_labels=BIN_LABELS):
    seqs = []
    levels = sorted(bin_labels.keys(), reverse=True)  # high→low
    for idx in X_disc.index:
        bins = X_disc.loc[idx]
        zvals = X_z.loc[idx]
        seq = []
        for lv in levels:
            feats = [f for f in X_disc.columns if bins[f] == lv]
            if not feats:
                continue
            feats.sort(key=lambda f: zvals[f], reverse=True)
            label = bin_labels[lv]
            itemset = {f"{f}_{label}" for f in feats}
            seq.append(itemset)
            if len(seq) >= top_k:
                break
        if not seq:
            best = zvals.abs().idxmax()
            label = bin_labels[int(bins[best])]
            seq = [{f"{best}_{label}"}]
        seqs.append(seq)
    return seqs

# ------------- small cache for speed -------------
_support_cache = {}

def seq_support(pattern, seqs):
    key = tuple(tuple(sorted(s)) for s in pattern)
    if key in _support_cache:
        return _support_cache[key]

    def has(seq, pat):
        i = 0
        for s in seq:
            if pat[i].issubset(s):
                i += 1
                if i == len(pat):
                    return True
        return False

    cnt = sum(1 for s in seqs if has(s, pattern))
    sup = cnt / max(1, len(seqs))
    _support_cache[key] = sup
    return sup

def make_candidates(prev):
    cands = []
    for a in prev:
        for b in prev:
            if a[:-1] == b[:-1] and a[-1] != b[-1]:
                c1 = a + [b[-1]]
                c2 = a[:-1] + [a[-1] | b[-1]]
                for c in (c1, c2):
                    if c not in cands:
                        cands.append(c)
    return cands

# ------------- gsp main -------------
def gsp(seqs, min_support=0.25, max_len=2):
    start = time.time()
    items = set(itertools.chain.from_iterable(itertools.chain.from_iterable(seqs)))
    log(f"GSP start: {len(seqs)} seqs, {len(items)} single items")
    C1 = [[{i}] for i in items]
    F = {}
    F1 = [(p, seq_support(p, seqs)) for p in C1 if seq_support(p, seqs) >= min_support]
    if not F1:
        log("no 1-length patterns meet support")
        return F
    F[1] = F1
    prev = [p for p, _ in F1]
    k = 1
    while k < max_len and prev:
        log(f"GSP level {k+1} start, {len(prev)} frequent patterns")
        Ck1 = make_candidates(prev)
        uniq = []
        seen = set()
        for c in Ck1:
            key = tuple(tuple(sorted(s)) for s in c)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        Fk1 = [(p, seq_support(p, seqs)) for p in uniq if seq_support(p, seqs) >= min_support]
        if not Fk1:
            break
        F[k+1] = Fk1
        prev = [p for p, _ in Fk1]
        log(f"GSP level {k+1} done ({len(Fk1)} patterns)")
        k += 1
    log(f"GSP done in {time.time()-start:.1f}s")
    return F

def pat_to_str(pat):
    return "<" + ", ".join("{" + ", ".join(sorted(s)) + "}" for s in pat) + ">"

# ------------- main pipeline -------------
def main():
    log("start task 2 script")
    safe_mkdir(RESULTS_DIR)

    log("loading csv")
    df = pd.read_csv(CSV_PATH)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    log("cleaning data")
    df[TARGET_COL] = encode_target(df[TARGET_COL])
    for c in df.columns:
        if c != TARGET_COL and not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    feats = [c for c in df.columns if c != TARGET_COL]
    X = df[feats]
    Xz = zscore(X)
    Xd = discretize(X, n_bins=N_BINS, strategy="quantile")

    log("building sequences")
    seqs = build_sequences(Xd, Xz, top_k=TOP_K)
    labels = df[TARGET_COL].values
    seq_m = [s for s, y in zip(seqs, labels) if y == 1]
    seq_b = [s for s, y in zip(seqs, labels) if y == 0]

    log("running gsp for Malignant")
    Fm = gsp(seq_m, min_support=MIN_SUPPORT, max_len=MAX_LEN)
    log("running gsp for Benign")
    Fb = gsp(seq_b, min_support=MIN_SUPPORT, max_len=MAX_LEN)

    log("saving results")
    rows = []
    for lbl, F in [("Malignant", Fm), ("Benign", Fb)]:
        for L, plist in F.items():
            for p, sup in plist:
                rows.append({
                    "class": lbl,
                    "len": L,
                    "pattern": pat_to_str(p),
                    "support": round(sup, 3)
                })
    out = pd.DataFrame(rows).sort_values(["class", "len", "support"], ascending=[True, True, False])
    out.to_csv(os.path.join(RESULTS_DIR, "task2_patterns.csv"), index=False)

    log("top 5 patterns per class:")
    for lbl in ["Malignant", "Benign"]:
        print(f"\n[{lbl}]")
        top = out[out["class"] == lbl].head(5)
        if top.empty:
            print("no pattern found")
        else:
            for _, r in top.iterrows():
                print(f" len={r['len']} | sup={r['support']} | {r['pattern']}")
    log("done all, file saved under task2_results/task2_patterns.csv")

if __name__ == "__main__":
    main()
