# train_qwen_pii_classifier_cfg.py
# Fine-tune Qwen2.5-VL-7B-Instruct (LoRA) for image-level PII classification (no boxes)
# No CLI flags needed — edit the CONFIG block and run:  python train_qwen_pii_classifier_cfg.py

import os, json, random, math, inspect
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import AutoProcessor, TrainingArguments, Trainer
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls  # fallback

from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# =========================
# CONFIG — EDIT THESE
# =========================
JSONL_PATH = "pii_42k.jsonl"
OUTDIR     = "runs\\qwen_pii_12k_lora"

EPOCHS = 8
PER_DEVICE_BATCH = 2
ACCUM = 8                      # global batch = PER_DEVICE_BATCH * ACCUM
LR_LORA = 2e-4
LR_PROJ = 1e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 25
SAVE_STEPS = None                 # None = auto; or set int (e.g., 500)
EVAL_STEPS = None                 # None = auto; or set int (e.g., 500)

# Image long-side cap (px)
LONG_SIDE_MAX = 896

# (Small smoke test; bump later)
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
    processor: AutoProcessor
    pad_token_id: int  # not used for length; we rely on attention masks

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images  = [b["image"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        answers = [b["answer"] for b in batch]

        # Build chat templates so Qwen gets <image> placeholders
        user_msgs  = [[{"role":"user","content":[{"type":"image"},{"type":"text","text": p}]}] for p in prompts]
        full_msgs  = [m + [{"role":"assistant","content":[{"type":"text","text": a}]}]
                      for m, a in zip(user_msgs, answers)]

        chat_user  = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                      for m in user_msgs]
        chat_full  = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                      for m in full_msgs]

        # 1) Prompt-only encoding (to compute mask length)
        enc_prompt = self.processor(text=chat_user, images=images, return_tensors="pt", padding=True)
        prompt_attn = enc_prompt["attention_mask"]

        # 2) Full (prompt+answer) encoding
        enc_full = self.processor(text=chat_full, images=images, return_tensors="pt", padding=True)

        input_ids       = enc_full["input_ids"]
        attention_mask  = enc_full["attention_mask"]
        pixel_values    = enc_full["pixel_values"]

        # --- FIX: don't use `or` with tensors ---
        image_grid_thw = enc_full.get("image_grid_thw", None)
        if image_grid_thw is None:
            image_grid_thw = enc_full.get("image_grid_thw_list", None)
        if image_grid_thw is None:
            enc_img = self.processor(images=images, return_tensors="pt")
            image_grid_thw = enc_img.get("image_grid_thw", None)
            if image_grid_thw is None:
                image_grid_thw = enc_img.get("image_grid_thw_list", None)

        # Labels: mask prompt tokens + padding
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

def make_splits(jsonl_path: str):
    sens, nons = [], []
    for rec in read_jsonl(jsonl_path):
        if not os.path.exists(rec["image"]): continue
        (sens if rec.get("is_sensitive", False) else nons).append(rec)
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
    try:
        l = s.find("{"); r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            obj = json.loads(s[l:r+1])
            lbls = obj.get("labels", {})
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

def main():
    assert os.path.isfile(JSONL_PATH), f"JSONL not found: {JSONL_PATH}"
    os.makedirs(OUTDIR, exist_ok=True)

    # speed knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(SEED)

    print("Loading & sampling dataset...")
    train_recs, val_recs, test_recs = make_splits(JSONL_PATH)
    print(f"Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}")

    print("Loading model & processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    PAD_ID = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    model = AutoModelCls.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # LoRA targets
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = PIIDataset(train_recs)
    val_ds   = PIIDataset(val_recs)
    collator = Collator(processor=processor, pad_token_id=PAD_ID)

    steps_per_epoch = math.ceil(len(train_ds) / (PER_DEVICE_BATCH * ACCUM))
    eval_steps = EVAL_STEPS if EVAL_STEPS is not None else max(50, steps_per_epoch)
    save_steps = SAVE_STEPS if SAVE_STEPS is not None else max(100, steps_per_epoch)

    training_args = TrainingArguments(
        output_dir=OUTDIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=ACCUM,
        per_device_eval_batch_size=1,
        bf16=(device=="cuda"),
        learning_rate=LR_LORA,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_steps=eval_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        report_to="none"
    )

    # Parameter groups (projector with different LR)
    proj_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ["multi_modal_projector","mm_projector","vision_proj"]):
            proj_params.append(p)
        else:
            other_params.append(p)
    optim_groups = []
    if other_params: optim_groups.append({"params": other_params, "lr": LR_LORA})
    if proj_params:  optim_groups.append({"params": proj_params, "lr": LR_PROJ})

    class MyTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                extra = {}
                sig = inspect.signature(torch.optim.AdamW)
                if "fused" in sig.parameters and torch.cuda.is_available():
                    extra["fused"] = True
                self.optimizer = torch.optim.AdamW(
                    optim_groups if optim_groups else self.model.parameters(),
                    lr=training_args.learning_rate,
                    betas=(0.9, 0.95), eps=1e-8, weight_decay=training_args.weight_decay,
                    **extra
                )
            return self.optimizer

    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Starting training...")
    trainer.train()

    trainer.save_model(OUTDIR)
    processor.save_pretrained(OUTDIR)

    # ---------- Evaluation on test ----------
    print("Evaluating on test split...")
    model.eval()
    test_ds = PIIDataset(test_recs)
    y_true, y_pred = [], []
    gen_kwargs = dict(max_new_tokens=128, do_sample=False, temperature=0.0, pad_token_id=PAD_ID)

    for rec in test_ds:
        im = rec["image"]
        im = cap_long_side(im, LONG_SIDE_MAX)
        W, H = im.size
        prompt = build_prompt(PII_TYPES, W, H)

        user_msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text": prompt}]}]
        chat_text = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[chat_text], images=[im], return_tensors="pt", padding=True)
        inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        pred = parse_json_pred(text)

        _, truth_labels = normalize_labels({
            "is_sensitive": rec["is_sensitive"],
            "types": [k for k,v in rec["labels_dict"].items() if v]
        })
        y_true.append(truth_labels)
        y_pred.append(pred)

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