#!/usr/bin/env python3
import os, json, random, shutil, subprocess, time, re, argparse, sys
from pathlib import Path

BIN_LINE_RE = re.compile(r"Binary \(any-PII\)\s*(?:->|→)\s*F1=([0-9.]+)\s+P=([0-9.]+)\s+R=([0-9.]+)\s+Acc=([0-9.]+)")

def is_sensitive(labels_dict):
    # Sensitive if any class is True
    return any(bool(v) for v in labels_dict.values())

def make_mixed_subset(src_jsonl: Path, dst_jsonl: Path, n_pos: int, n_neg: int, seed: int = 42):
    rng = random.Random(seed)
    pos, neg = [], []
    with src_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            lab = obj.get("labels", {})
            (pos if is_sensitive(lab) else neg).append(obj)

    if len(pos) < n_pos or len(neg) < n_neg:
        raise RuntimeError(f"Not enough samples: have {len(pos)} positives and {len(neg)} negatives, "
                           f"but need {n_pos}/{n_neg}.")

    pos_s = rng.sample(pos, n_pos)
    neg_s = rng.sample(neg, n_neg)
    mix = pos_s + neg_s
    rng.shuffle(mix)

    with dst_jsonl.open("w", encoding="utf-8") as out:
        for obj in mix:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(mix)

def run_eval(log_path: Path, adapter_path: str | None):
    env = os.environ.copy()
    env.update({
        "TRANSFORMERS_NO_TF": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "EVAL_ONLY": "1",
        # We only care about evaluating the current JSONL as-is
        "SPLIT": "test",
    })
    # IMPORTANT: make sure ADAPTER_PATH is unset for the Base run
    if adapter_path is not None:
        env["ADAPTER_PATH"] = adapter_path
    else:
        env.pop("ADAPTER_PATH", None)

    proc = subprocess.run(
        ["python", "t.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, proc.stdout

def count_items_and_binary_metrics(log_text: str):
    # Count evaluated items by instances of PRED_POS/GT_POS pairs
    pairs = []
    cur = {"pred": None, "gt": None}
    for line in log_text.splitlines():
        mp = re.search(r"PRED_POS=\[(.*)\]", line)
        if mp:
            txt = mp.group(1).strip()
            pred = [] if txt == "" else [s.strip().strip("'\"") for s in txt.split(",") if s.strip()]
            cur["pred"] = pred
        mg = re.search(r"GT_POS=\[(.*)\]", line)
        if mg:
            txt = mg.group(1).strip()
            gt = [] if txt == "" else [s.strip().strip("'\"") for s in txt.split(",") if s.strip()]
            cur["gt"] = gt
            if cur["pred"] is not None:
                pairs.append(cur)
                cur = {"pred": None, "gt": None}

    n = len(pairs)

    # Try to read the Binary(any-PII) line first
    m = BIN_LINE_RE.search(log_text)
    if m:
        f1, p, r, acc = map(float, m.groups())
        return n, {"precision": p, "recall": r, "f1": f1, "acc": acc}

    # Fallback: compute from pairs
    tp = fp = fn = tn = 0
    for pr in pairs:
        pred_pos = len(pr["pred"]) > 0
        gt_pos = len(pr["gt"]) > 0
        if pred_pos and gt_pos: tp += 1
        elif pred_pos and not gt_pos: fp += 1
        elif not pred_pos and gt_pos: fn += 1
        else: tn += 1

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return n, {"precision": p, "recall": r, "f1": f1, "acc": acc}

def main():
    ap = argparse.ArgumentParser(description="Compare Base Qwen2-VL vs LoRA checkpoint on a mixed subset (binary any-PII).")
    ap.add_argument("--src-jsonl", default="/notebooks/pii_42k.jsonl", type=Path)
    ap.add_argument("--root", default="/notebooks", type=Path, help="Project root (contains t.py).")
    ap.add_argument("--adapter-path", default="runs/qwen_pii_12k_lora/checkpoint-4000")
    ap.add_argument("--n-pos", type=int, default=500)
    ap.add_argument("--n-neg", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, default=Path("runs/compare_mixed"))
    args = ap.parse_args()

    os.chdir(args.root)

    src = args.src_jsonl.resolve()
    tmp = src.parent / "mixed_eval.jsonl"
    bak = src.parent / f"{src.name}.bak_{int(time.time())}"

    print(f"[1/5] Creating mixed subset → {tmp}")
    total = make_mixed_subset(src, tmp, args.n_pos, args.n_neg, args.seed)
    print(f"    Mixed subset size: {total} images ({args.n_pos} pos + {args.n_neg} neg)")

    print(f"[2/5] Swapping JSONL (backup: {bak.name})")
    shutil.copy2(src, bak)
    shutil.move(tmp, src)

    try:
        base_log = args.outdir / "base_on_mixed.log"
        lora_log = args.outdir / "lora_ckpt4000_on_mixed.log"

        print(f"[3/5] Evaluating **Base Qwen2-VL** (no adapter)…")
        rc_base, out_base = run_eval(base_log, adapter_path=None)
        n_base, m_base = count_items_and_binary_metrics(out_base)
        print(f"    Done. N={n_base} | Binary(any-PII) F1={m_base['f1']:.4f}, P={m_base['precision']:.4f}, R={m_base['recall']:.4f}, Acc={m_base['acc']:.4f}")

        print(f"[4/5] Evaluating **LoRA checkpoint-4000**…")
        rc_lora, out_lora = run_eval(lora_log, adapter_path=args.adapter_path)
        n_lora, m_lora = count_items_and_binary_metrics(out_lora)
        print(f"    Done. N={n_lora} | Binary(any-PII) F1={m_lora['f1']:.4f}, P={m_lora['precision']:.4f}, R={m_lora['recall']:.4f}, Acc={m_lora['acc']:.4f}")

    finally:
        print(f"[5/5] Restoring original JSONL")
        if src.exists():
            src.unlink()
        shutil.move(bak, src)

    # Final comparison table
    print("\n===== COMPARISON (Binary any-PII on mixed subset) =====")
    print(f"{'Model':28} {'N':>6} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    print("-" * 64)
    print(f"{'Base Qwen2-VL':28} {n_base:6d} {m_base['f1']:8.4f} {m_base['precision']:8.4f} {m_base['recall']:8.4f} {m_base['acc']:8.4f}")
    print(f"{'LoRA (ckpt-4000)':28} {n_lora:6d} {m_lora['f1']:8.4f} {m_lora['precision']:8.4f} {m_lora['recall']:8.4f} {m_lora['acc']:8.4f}")
    print("=======================================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
