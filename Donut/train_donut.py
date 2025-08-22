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

# Set decoder max length
model.config.decoder.max_length = MAX_LENGTH
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Freeze encoder (optional for lower VRAM)
# model.encoder.requires_grad_(False)

# === Load and Preprocess Dataset ===
print("📄 Loading annotations...")
with open(ANNOTATIONS_PATH, "r") as f:
    raw_data = json.load(f)

# Split into train/val (90/10)
split_idx = int(len(raw_data) * 0.9)
train_data = raw_data[:split_idx]
val_data = raw_data[split_idx:]

# -------------------------------------------------------------------------- #
# Debug: Limit to 500 examples
train_data = train_data[:500]
val_data = val_data[:50]  # Optional: also reduce validation for speed
# -------------------------------------------------------------------------- #

# Preprocessing function
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values[0]

    task_prompt = example["task_prompt"]
    target_text = json.dumps(example["ground_truth"], separators=(",", ":"))

    input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    labels = processor.tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore padding in loss

    return {
        "pixel_values": pixel_values.numpy(),   
        "input_ids": input_ids.tolist(),          
        "labels": labels.tolist()                 
    }

# Convert to Hugging Face Dataset
print("📦 Processing dataset...")
train_ds = Dataset.from_list(train_data).map(preprocess)
val_ds = Dataset.from_list(val_data).map(preprocess)

# === Training Arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=LOG_DIR,
    save_total_limit=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-5,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    report_to="none"  # set to "wandb" or "tensorboard" if needed
)

# === Initialize Trainer ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=processor.tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator
)

# === Start Training ===
print("🚀 Starting training...")
trainer.train()

print("✅ Done training! Model saved to:", OUTPUT_DIR)
