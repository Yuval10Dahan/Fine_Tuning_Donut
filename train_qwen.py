# t.py
# Fine-tune Qwen2.5-VL-7B-Instruct (LoRA) for image-level PII classification (no boxes)
# Supports EVAL_ONLY=1 to skip training and just run evaluation with a trained LoRA.

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import json, random, math, inspect, urllib.request, zipfile, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from packaging import version

from transformers import TrainingArguments, Trainer
try:
    from transformers import Qwen2VLProcessor as ProcessorCls
except Exception:
    from transformers import AutoProcessor as ProcessorCls

try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls  # fallback

from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

# =========================
# CONFIG — EDIT THESE
# =========================
USE_DROPBOX      = True
DROPBOX_ZIP_URL  = "https://www.dropbox.com/scl/fi/48brdm5zhu5mvqxwt512d/datasets.zip?rlkey=vv99353pto1xk9u0ssyaykgbj&dl=1"
ZIP_LOCAL        = "datasets.zip"
EXTRACT_DIR      = "datasets"                 # where the zip will be extracted

# Prefer a JSONL next to this script, else we’ll search EXTRACT_DIR recursively.
JSONL_FILENAME   = "pii_42k.jsonl"

OUTDIR           = "runs/qwen_pii_12k_lora"

# Training config (will be auto-dialed down on CPU)
EPOCHS = 8
PER_DEVICE_BATCH = 2
ACCUM = 8                                   # global batch = PER_DEVICE_BATCH * ACCUM
LR_LORA = 2e-4
LR_PROJ = 1e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 25
SAVE_STEPS = None                           # None = auto; or set int
EVAL_STEPS = None                           # None = auto

LONG_SIDE_MAX = 896

# Dataset sizes (adjust to your needs)
TRAIN_SENSITIVE = 4000
TRAIN_NON_SENSITIVE = 4000
VAL_SENSITIVE   = 1000
VAL_NON_SENSITIVE = 1000
TEST_SENSITIVE  = 1000
TEST_NON_SENSITIVE = 1000
# =========================

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SEED = 42

PII_TYPES = [
    "CREDIT_CARD_NUMBER","SSN","DRIVER_LICENSE","PERSONAL_ID",
    "PIN_CODE","MEDICAL_LETTER","PHONE_BILL","NAME","ADDRESS",
    "EMAIL","PHONE","OTHER_PII","BANK_ACCOUNT_NUMBER",
]

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def env_flag(name: str) -> bool:
    v = os.environ.get(name, "")
    return str(v).strip() not in ("", "0", "false", "False", "no", "No")

EVAL_ONLY  = env_flag("EVAL_ONLY")   # if set, skip training and only evaluate
DEBUG_EVAL = env_flag("DEBUG_EVAL")  # print per-sample debug during eval

def download_if_needed(url: str, out_zip: str):
    if os.path.isfile(out_zip):
        print(f"[Dropbox] ZIP already exists: {out_zip}")
        return
    print(f"[Dropbox] Downloading ZIP from: {url}")
    urllib.request.urlretrieve(url, out_zip)
    print(f"[Dropbox] Saved to: {out_zip}")

def extract_zip(zip_path: str, extract_to: str) -> None:
    os.makedirs(extract_to, exist_ok=True)
    print(f"[Dropbox] Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print("[Dropbox] Extraction complete.")

def resolve_jsonl_path(jsonl_filename: str, extract_dir: str) -> str:
    here = Path(__file__).parent.resolve()
    local = here / jsonl_filename
    if local.is_file():
        print(f"[JSONL] Using local JSONL: {local}")
        return str(local)
    for p in Path(extract_dir).rglob(jsonl_filename):
        print(f"[JSONL] Found JSONL in extract dir: {p}")
        return str(p)
    any_jsonl = list(Path(extract_dir).rglob("*.jsonl"))
    if any_jsonl:
        print(f"[JSONL][WARN] Using first JSONL found in extract dir: {any_jsonl[0]}")
        return str(any_jsonl[0])
    raise FileNotFoundError(f"Could not find {jsonl_filename} locally or inside {extract_dir}")

def locate_dataset_root(extract_dir: str) -> str:
    for p in Path(extract_dir).rglob("pii"):
        if (p / "sensitive").is_dir() and (p / "non_sensitive").is_dir():
            print(f"[Root] Using dataset root: {p}")
            return str(p.resolve())
    for p in Path(extract_dir).rglob("*"):
        if p.is_dir() and (p / "sensitive").is_dir() and (p / "non_sensitive").is_dir():
            print(f"[Root] Using dataset root: {p}")
            return str(p.resolve())
    print(f"[Root][WARN] Could not find 'pii' with subfolders; using: {extract_dir}")
    return str(Path(extract_dir).resolve())

def build_basename_index(root_with_images: str) -> Dict[str, str]:
    idx = {}
    for p in Path(root_with_images).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.name, str(p.resolve()))
    print(f"[PathFix] Indexed {len(idx)} image basenames under {root_with_images}")
    return idx

def repair_image_path(original_path: str, dataset_root: str,
                      basename_index: Optional[Dict[str,str]] = None) -> Optional[str]:
    if os.path.exists(original_path):
        return original_path
    p = original_path.replace("\\", "/").lower()
    for anchor in ("sensitive/", "non_sensitive/"):
        pos = p.find(anchor)
        if pos != -1:
            rel = original_path.replace("\\", "/")[pos:]
            candidate = os.path.join(dataset_root, rel)
            if os.path.exists(candidate):
                return candidate
    if basename_index:
        base = os.path.basename(original_path)
        cand = basename_index.get(base)
        if cand and os.path.exists(cand):
            return cand
    return None

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def normalize_labels(rec: Dict[str, Any]):
    is_sensitive = bool(rec.get("is_sensitive", False))
    types = rec.get("types", []) or []
    types = [t for t in types if t in PII_TYPES]
    labels = {t: False for t in PII_TYPES}
    for t in types:
        labels[t] = True
    return is_sensitive, labels

def cap_long_side(im: Image.Image, long_side_max=LONG_SIDE_MAX):
    w, h = im.size
    m = max(w, h)
    if m <= long_side_max: return im
    scale = long_side_max / float(m)
    return im.resize((int(round(w*scale)), int(round(h*scale))), Image.BICUBIC)

def build_prompt(class_list: List[str], width: int, height: int) -> str:
    return (
        "You are a PII auditor. Read the image and decide which PII types are present. "
        "Consider these classes ONLY:\n"
        + ", ".join(class_list) + ".\n"
        "Return ONLY JSON with fields:\n"
        "{\n"
        '  "labels": {<CLASS>: true|false, ...},\n'
        '  "evidence_text": ["short snippets that justify positives"]\n'
        "}\n"
        f"Image size (pixels): width={width}, height={height}."
    )

def build_answer_json(is_sensitive: bool, labels_dict: Dict[str, bool]) -> str:
    return json.dumps({"labels": labels_dict, "evidence_text": []}, ensure_ascii=False)

class PIIDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.recs = records
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        r = self.recs[idx]
        path = r["image"]
        im = Image.open(path).convert("RGB")
        im = cap_long_side(im, LONG_SIDE_MAX)
        W, H = im.size
        is_sensitive, labels = normalize_labels(r)
        prompt = build_prompt(PII_TYPES, W, H)
        answer = build_answer_json(is_sensitive, labels)
        return {"image": im, "prompt": prompt, "answer": answer,
                "is_sensitive": is_sensitive, "labels_dict": labels,
                "image_path": path}

@dataclass
class Collator:
    processor: ProcessorCls
    pad_token_id: int
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images  = [b["image"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        answers = [b["answer"] for b in batch]
        user_msgs  = [[{"role":"user","content":[{"type":"image"},{"type":"text","text": p}]}] for p in prompts]
        full_msgs  = [m + [{"role":"assistant","content":[{"type":"text","text": a}]}]
                      for m, a in zip(user_msgs, answers)]
        chat_user  = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                      for m in user_msgs]
        chat_full  = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                      for m in full_msgs]
        enc_prompt = self.processor(text=chat_user, images=images, return_tensors="pt", padding=True)
        prompt_attn = enc_prompt["attention_mask"]
        enc_full = self.processor(text=chat_full, images=images, return_tensors="pt", padding=True)
        input_ids       = enc_full["input_ids"]
        attention_mask  = enc_full["attention_mask"]
        pixel_values    = enc_full["pixel_values"]
        image_grid_thw = enc_full.get("image_grid_thw", None) or enc_full.get("image_grid_thw_list", None)
        if image_grid_thw is None:
            enc_img = self.processor(images=images, return_tensors="pt")
            image_grid_thw = enc_img.get("image_grid_thw", None) or enc_img.get("image_grid_thw_list", None)
        labels = input_ids.clone()
        for i in range(len(batch)):
            prompt_len = int(prompt_attn[i].sum().item())
            labels[i, :prompt_len] = -100
            labels[i, attention_mask[i] == 0] = -100
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }
        if image_grid_thw is not None:
            out["image_grid_thw"] = image_grid_thw
        return out

def make_splits(jsonl_path: str, dataset_root: str):
    basename_index = None
    sens, nons = [], []
    missing = 0
    for rec in read_jsonl(jsonl_path):
        img_path = rec.get("image", "")
        if not os.path.exists(img_path):
            if basename_index is None:
                basename_index = build_basename_index(dataset_root)
            fixed = repair_image_path(img_path, dataset_root, basename_index)
            if fixed is None:
                missing += 1
                continue
            rec["image"] = fixed
        (sens if rec.get("is_sensitive", False) else nons).append(rec)
    if missing > 0:
        print(f"[PathFix] Skipped {missing} records with unresolved image paths.")
    random.shuffle(sens); random.shuffle(nons)
    need_s = TRAIN_SENSITIVE + VAL_SENSITIVE + TEST_SENSITIVE
    need_n = TRAIN_NON_SENSITIVE + VAL_NON_SENSITIVE + TEST_NON_SENSITIVE
    assert len(sens) >= need_s, f"Not enough sensitive images: {len(sens)} < {need_s}"
    assert len(nons) >= need_n, f"Not enough non-sensitive images: {len(nons)} < {need_n}"
    train = sens[:TRAIN_SENSITIVE] + nons[:TRAIN_NON_SENSITIVE]
    val   = sens[TRAIN_SENSITIVE:TRAIN_SENSITIVE+VAL_SENSITIVE] + \
            nons[TRAIN_NON_SENSITIVE:TRAIN_NON_SENSITIVE+VAL_NON_SENSITIVE]
    test  = sens[TRAIN_SENSITIVE+VAL_SENSITIVE:TRAIN_SENSITIVE+VAL_SENSITIVE+TEST_SENSITIVE] + \
            nons[TRAIN_NON_SENSITIVE+VAL_NON_SENSITIVE:
                 TRAIN_NON_SENSITIVE+VAL_NON_SENSITIVE+TEST_NON_SENSITIVE]
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test

def parse_json_pred(s: str) -> Dict[str, bool]:
    # Strip code fences if present
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n", "", s)  # remove ```json\n
        s = re.sub(r"\n```$", "", s)
    # Extract first JSON object
    try:
        l = s.find("{"); r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            obj = json.loads(s[l:r+1])
            lbls = obj.get("labels", {}) or {}
            return {k: bool(lbls.get(k, False)) for k in PII_TYPES}
    except Exception:
        pass
    return {k: False for k in PII_TYPES}

def metrics_from_preds(y_true: List[Dict[str,bool]], y_pred: List[Dict[str,bool]]):
    per = {}
    for cls in PII_TYPES:
        t = [int(x.get(cls, False)) for x in y_true]
        p = [int(x.get(cls, False)) for x in y_pred]
        if sum(t) == 0 and sum(p) == 0: continue
        per[cls] = {
            "f1": f1_score(t, p, zero_division=0),
            "precision": precision_score(t, p, zero_division=0),
            "recall": recall_score(t, p, zero_division=0),
            "support": int(sum(t)),
        }
    macro_f1 = sum(v["f1"] for v in per.values()) / max(len(per), 1)
    t_bin = [int(any(d.values())) for d in y_true]
    p_bin = [int(any(d.values())) for d in y_pred]
    bin_metrics = {
        "f1": f1_score(t_bin, p_bin, zero_division=0),
        "precision": precision_score(t_bin, p_bin, zero_division=0),
        "recall": recall_score(t_bin, p_bin, zero_division=0),
        "accuracy": accuracy_score(t_bin, p_bin),
    }
    return {"per_class": per, "macro_f1": macro_f1, "binary": bin_metrics}

def _latest_ckpt_dir(outdir: str) -> Optional[str]:
    p = Path(outdir)
    if not p.exists(): return None
    cks = [d for d in p.glob("checkpoint-*") if d.is_dir()]
    if not cks: return None
    return str(max(cks, key=lambda d: d.stat().st_mtime))

def _strip_state_files_for_torch_lt_26(ckpt_dir: str):
    for fname in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"):
        f = os.path.join(ckpt_dir, fname)
        if os.path.exists(f):
            os.remove(f)

def main():
    if USE_DROPBOX:
        download_if_needed(DROPBOX_ZIP_URL, ZIP_LOCAL)
        extract_zip(ZIP_LOCAL, EXTRACT_DIR)

    jsonl_path = resolve_jsonl_path(JSONL_FILENAME, EXTRACT_DIR)
    dataset_root = locate_dataset_root(EXTRACT_DIR)

    os.makedirs(OUTDIR, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(SEED)

    print("Loading & sampling dataset...")
    train_recs, val_recs, test_recs = make_splits(jsonl_path, dataset_root)
    print(f"Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}")

    # --------- Device & run mode ----------
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = "cuda"
        dtype = torch.bfloat16
        run_mode = "FULL"
        do_eval = not EVAL_ONLY  # during training mode, eval val each steps
        max_steps_override = None
        per_device_train_batch = PER_DEVICE_BATCH
        grad_accum = ACCUM
        optim_name = "adamw_torch_fused"
        num_workers = 0
        pin_mem = False
    else:
        device = "cpu"
        dtype = torch.float32
        run_mode = "CPU_SMOKE_TEST"
        do_eval = False
        max_steps_override = 1
        per_device_train_batch = 1
        grad_accum = 1
        optim_name = "adamw_torch"
        num_workers = 0
        pin_mem = False
        if not EVAL_ONLY:
            print("\n[CPU] CUDA not available — running a quick CPU smoke test (1 train step, no eval).\n")

    print("Loading model & processor...")
    processor = ProcessorCls.from_pretrained(MODEL_ID, trust_remote_code=True)
    PAD_ID = getattr(processor.tokenizer, "pad_token_id", None) or processor.tokenizer.eos_token_id

    # ===== Model & (maybe) adapters =====
    base_model = AutoModelCls.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True)
    base_model.to(device)
    base_model.config.use_cache = False  # off during train; will re-enable for eval

    # Figure out adapter directory (latest checkpoint preferred)
    latest = _latest_ckpt_dir(OUTDIR)
    adapter_dir = latest if latest else OUTDIR

    if EVAL_ONLY:
        # Do NOT wrap the model again; just attach the trained adapter for inference.
        if os.path.isdir(adapter_dir):
            model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
            print(f"[Eval-Only] Loaded adapters from {adapter_dir}")
        else:
            model = base_model  # fall back (will evaluate base model)
            print("[Eval-Only][WARN] No adapter directory found; evaluating base model.")
    else:
        # Training path: wrap with LoRA and (if resuming) Trainer will load state.
        if has_cuda:
            try:
                base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base_model.gradient_checkpointing_enable()
            if hasattr(base_model, "enable_input_require_grads"):
                base_model.enable_input_require_grads()
        lora_targets = [
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
            "wo","wq","wk","wv",
            "multi_modal_projector","mm_projector","vision_proj"
        ]
        peft_config = LoraConfig(
            r=32, lora_alpha=32, lora_dropout=0.05,
            target_modules=lora_targets, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()

    train_ds = PIIDataset(train_recs)
    val_ds   = PIIDataset(val_recs)
    collator = Collator(processor=processor, pad_token_id=PAD_ID)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (per_device_train_batch * grad_accum)))
    eval_steps = (EVAL_STEPS if (do_eval and EVAL_STEPS is not None) else
                  (max(50, steps_per_epoch) if do_eval else None))
    save_steps = SAVE_STEPS if SAVE_STEPS is not None else max(100, steps_per_epoch)

    training_args = TrainingArguments(
        output_dir=OUTDIR,
        num_train_epochs=float(EPOCHS),
        max_steps=(max_steps_override if max_steps_override is not None else -1),

        per_device_train_batch_size=per_device_train_batch,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=1,

        bf16=has_cuda,

        learning_rate=LR_LORA,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        logging_steps=LOGGING_STEPS,
        eval_steps=eval_steps,
        save_steps=save_steps,
        eval_strategy=("steps" if do_eval else "no"),
        save_strategy="steps",
        save_total_limit=2,
        save_safetensors=True,

        remove_unused_columns=False,
        optim=("adamw_torch_fused" if has_cuda else "adamw_torch"),
        report_to="none",

        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=pin_mem,
        **({"dataloader_prefetch_factor": 2} if num_workers > 0 else {})
    )

    class MyTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                extra = {}
                sig = inspect.signature(torch.optim.AdamW)
                if "fused" in sig.parameters and torch.cuda.is_available():
                    extra["fused"] = True
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=training_args.learning_rate,
                    betas=(0.9, 0.95), eps=1e-8, weight_decay=training_args.weight_decay,
                    **extra
                )
            return self.optimizer

    print("CUDA:", torch.cuda.is_available(), "| Device:", device)
    print("Dataset root:", dataset_root)
    print("JSONL path  :", jsonl_path)
    print("Run mode    :", ("EVAL_ONLY" if EVAL_ONLY else run_mode))
    print("Steps/epoch :", steps_per_epoch)

    # ===== Train (unless EVAL_ONLY) =====
    if not EVAL_ONLY:
        trainer = MyTrainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_ds,
            eval_dataset=(val_ds if do_eval else None),
        )
        print("Starting training...")
        latest_ckpt = _latest_ckpt_dir(OUTDIR)
        if latest_ckpt:
            if version.parse(torch.__version__) < version.parse("2.6.0"):
                _strip_state_files_for_torch_lt_26(latest_ckpt)
                print(f"[Resume] torch {torch.__version__} < 2.6 — resuming from {latest_ckpt} WITHOUT optimizer/scheduler/RNG state.")
            else:
                print(f"[Resume] Resuming from {latest_ckpt} with full state.")
            trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            trainer.train()

        trainer.save_model(OUTDIR)
        processor.save_pretrained(OUTDIR)

    # ===== Evaluation (test split) =====
    print("Evaluating on test split...")
    model.eval()
    model.config.use_cache = True  # better generation speed

    test_ds = PIIDataset(test_recs)
    y_true, y_pred = [], []

    gen_kwargs = dict(
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=PAD_ID,
        eos_token_id=getattr(getattr(model, "generation_config", None), "eos_token_id", processor.tokenizer.eos_token_id),
    )

    for i, rec in enumerate(test_ds):
        im = rec["image"]
        W, H = im.size
        prompt = build_prompt(PII_TYPES, W, H)

        user_msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text": prompt}]}]
        chat_text = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[chat_text], images=[im], return_tensors="pt", padding=True)
        inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        # Decode ONLY the generated continuation (not the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[:, prompt_len:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        pred = parse_json_pred(text)

        _, truth_labels = normalize_labels({
            "is_sensitive": rec["is_sensitive"],
            "types": [k for k,v in rec["labels_dict"].items() if v]
        })
        y_true.append(truth_labels)
        y_pred.append(pred)

        if DEBUG_EVAL and i < 50:
            pred_pos = [k for k,v in pred.items() if v]
            gt_pos   = [k for k,v in truth_labels.items() if v]
            shown = text
            if len(shown) > 300:
                shown = shown[:300] + "..."
            print(f"[{i}] TEXT={json.dumps(shown)}")
            print(f"     PRED_POS={pred_pos}")
            print(f"     GT_POS={gt_pos}")

    metrics = metrics_from_preds(y_true, y_pred)
    print("\n==== TEST METRICS ====")
    print(f"Macro-F1 (per-class): {metrics['macro_f1']:.4f}")
    print(f"Binary (any-PII)  ->  F1={metrics['binary']['f1']:.3f}  "
          f"P={metrics['binary']['precision']:.3f}  R={metrics['binary']['recall']:.3f}  "
          f"Acc={metrics['binary']['accuracy']:.3f}")
    for cls, m in metrics["per_class"].items():
        print(f"{cls:>18s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  (support={m['support']})")

    print("\nDone.")

if __name__ == "__main__":
    main()