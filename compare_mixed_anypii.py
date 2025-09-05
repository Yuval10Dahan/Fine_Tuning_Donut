#!/usr/bin/env python3
import os, re, shutil, subprocess, sys, time
from pathlib import Path

# parse Binary(any-PII) metrics from t.py output
BIN_LINE_RE = re.compile(r"Binary \(any-PII\)\s*(?:->|→)\s*F1=([0-9.]+)\s+P=([0-9.]+)\s+R=([0-9.]+)\s+Acc=([0-9.]+)")

def run_eval(log_path: Path, adapter_path: str | None, split: str):
    env = os.environ.copy()
    env.update({
        "TRANSFORMERS_NO_TF": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "EVAL_ONLY": "1",
        "SPLIT": split,  # 'test' or 'val' – whatever you want to evaluate
    })
    if adapter_path:
        env["ADAPTER_PATH"] = adapter_path
    else:
        env.pop("ADAPTER_PATH", None)

    proc = subprocess.run(
        ["python", "t.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=env,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.stdout

def count_items_and_metrics(log_text: str):
    # Count items by pairing PRED_POS / GT_POS lines
    n = 0
    saw_pred = False
    for line in log_text.splitlines():
        if "PRED_POS=[" in line: saw_pred = True
        if saw_pred and "GT_POS=[" in line:
            n += 1
            saw_pred = False

    m = BIN_LINE_RE.search(log_text)
    if m:
        f1, p, r, acc = map(float, m.groups())
        return n, f1, p, r, acc
    # fallback if Binary line not printed
    return n, float("nan"), float("nan"), float("nan"), float("nan")

def main():
    import argparse
    ap = argparse.ArgumentParser("Compare Base vs LoRA on a given JSONL (no mixing/sampling).")
    ap.add_argument("--project-root", default="/notebooks", type=Path, help="Folder with t.py")
    ap.add_argument("--jsonl", default="/notebooks/pii_3k.jsonl", type=Path, help="Your custom JSONL")
    ap.add_argument("--tpy-jsonl", default="/notebooks/pii_42k.jsonl", type=Path,
                    help="The JSONL path that t.py normally reads (will be temporarily replaced).")
    ap.add_argument("--adapter-path", default="runs/qwen_pii_12k_lora/checkpoint-4000")
    ap.add_argument("--split", default="test", choices=["test","val"], help="Which split t.py should evaluate")
    ap.add_argument("--outdir", default=Path("runs/compare_on_jsonl"), type=Path)
    args = ap.parse_args()

    os.chdir(args.project_root)
    src_jsonl = args.tpy_jsonl.resolve()
    custom_jsonl = args.jsonl.resolve()
    if not custom_jsonl.exists():
        print(f"ERROR: {custom_jsonl} not found.", file=sys.stderr)
        sys.exit(2)
    if not src_jsonl.exists():
        print(f"ERROR: Expected {src_jsonl} to exist (this is what t.py loads).", file=sys.stderr)
        sys.exit(2)

    # backup and swap
    bak = src_jsonl.parent / f"{src_jsonl.name}.bak_{int(time.time())}"
    print(f"[1/4] Backing up {src_jsonl} -> {bak}")
    shutil.copy2(src_jsonl, bak)
    print(f"[2/4] Swapping in your list -> {src_jsonl}")
    shutil.copy2(custom_jsonl, src_jsonl)

    try:
        base_log = args.outdir / f"base_on_{custom_jsonl.stem}.log"
        lora_log = args.outdir / f"lora4000_on_{custom_jsonl.stem}.log"

        print(f"[3/4] Evaluating BASE Qwen2-VL (no adapter)…")
        base_out = run_eval(base_log, adapter_path=None, split=args.split)
        n_b, f1_b, p_b, r_b, a_b = count_items_and_metrics(base_out)
        print(f"      N={n_b}  F1={f1_b:.4f}  P={p_b:.4f}  R={r_b:.4f}  Acc={a_b:.4f}")

        print(f"[3/4] Evaluating LoRA checkpoint-4000…")
        lora_out = run_eval(lora_log, adapter_path=args.adapter_path, split=args.split)
        n_l, f1_l, p_l, r_l, a_l = count_items_and_metrics(lora_out)
        print(f"      N={n_l}  F1={f1_l:.4f}  P={p_l:.4f}  R={r_l:.4f}  Acc={a_l:.4f}")

    finally:
        print(f"[4/4] Restoring original JSONL")
        shutil.move(str(bak), str(src_jsonl))

    print("\n===== COMPARISON (Binary any-PII) =====")
    print(f"{'Model':24} {'N':>6} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    print("-"*60)
    print(f"{'Base Qwen2-VL':24} {n_b:6d} {f1_b:8.4f} {p_b:8.4f} {r_b:8.4f} {a_b:8.4f}")
    print(f"{'LoRA (ckpt-4000)':24} {n_l:6d} {f1_l:8.4f} {p_l:8.4f} {r_l:8.4f} {a_l:8.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
