import json
import os
from PIL import Image
from datasets import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import torch

# === CONFIG ===
ANNOTATIONS_PATH = "data/annotations.json"
MODEL_NAME = "naver-clova-ix/donut-base"
OUTPUT_DIR = "./donut_sensitive_model"
LOG_DIR = "./logs"
MAX_LENGTH = 512

# === Load Donut Processor and Model ===
print("🔧 Loading Donut model and processor...")
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s>"])[0]
model.config.pad_token_id = processor.tokenizer.pad_token_id 


# Apply LoRA (LoRA setup not shown here)
print("🔧 Applying LoRA...")
# [Insert LoRA code here if you're using it]
print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)} || all params: {sum(p.numel() for p in model.parameters())} || trainable%: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.4f}")

# === Load and Preprocess Dataset ===
print("📄 Loading annotations...") 
with open(ANNOTATIONS_PATH, "r") as f:
    raw_data = json.load(f)

# Split into train/val (90/10)
split = int(0.9 * len(raw_data))
train_data = raw_data[:split]
val_data = raw_data[split:]

# ------------------------------------- #
# Debug: Limit to 500 examples
train_data = train_data[:500]
val_data = val_data[:50]
# ------------------------------------- #

print(f"📊 Training examples: {len(train_data)} | Validation examples: {len(val_data)}")

# === Preprocessing Function ===
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze(0)

    # 🔧 Convert the ground_truth dictionary to a string label
    target_text = example["task_prompt"] + json.dumps(example["ground_truth"], separators=(",", ":"))

    input_ids = processor.tokenizer(
        target_text,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt"
    ).input_ids.squeeze(0)

    return {
        "pixel_values": pixel_values,
        "labels": input_ids
    }

# === Convert to HuggingFace Datasets and Apply Mapping ===
print("📦 Processing dataset...")

train_ds = Dataset.from_list(train_data).map(
    preprocess,
    batched=False,
    writer_batch_size=50
)

val_ds = Dataset.from_list(val_data).map(
    preprocess,
    batched=False,
    writer_batch_size=50
)

# === Training Setup ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="steps", 
    logging_dir=LOG_DIR,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor.tokenizer,
    data_collator=default_data_collator,
)

# === Start Training ===
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete.")
