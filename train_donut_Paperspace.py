import warnings
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


# === DOWNLOAD AND EXTRACT ZIP FROM GOOGLE DRIVE ===
print("=== Downloading and extracting dataset from Google Drive ===\n")
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

file_id = "1EkVzaYIEjLpgEjgA1q-PeDUwXAboI_1V"  
zip_url = f"https://drive.google.com/uc?id={file_id}"

# Download ZIP
if not os.path.exists("data.zip"):
    gdown.download(zip_url, output="data.zip", quiet=False)

# Unzip contents
os.system("unzip -o data.zip -d ./")


# === CONFIG ===
ANNOTATIONS_PATH = "annotations.json"
MODEL_NAME = "naver-clova-ix/donut-base"
OUTPUT_DIR = "./donut_sensitive_model"
LOG_DIR = "./logs"
MAX_LENGTH = 512


# === Load Donut Processor and Model ===
print("=== Loading Donut model and processor === \n")
processor = DonutProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s>"])[0]
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.loss_type = "ForCausalLMLoss"  # silences loss warning


# === Load and Preprocess Dataset ===
print("=== Loading annotations === \n")
with open(ANNOTATIONS_PATH, "r") as f:
    raw_data = json.load(f)

# Split into train/val (90/10)
split = int(0.9 * len(raw_data))
train_data = raw_data[:split]
val_data = raw_data[split:]

print(f"Total examples: {len(raw_data)}")
print(f"Training examples: {len(train_data)} | Validation examples: {len(val_data)} \n")


# === Preprocessing Function ===
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze(0)
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


# === Convert to Dataset ===
print("=== Processing dataset === \n")
train_ds = Dataset.from_list(train_data).map(preprocess, batched=False, writer_batch_size=50)
val_ds = Dataset.from_list(val_data).map(preprocess, batched=False, writer_batch_size=50)


# === Training Arguments ===
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


# === Initialize Trainer ===
print("=== Initializing trainer ===\n")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor,
    data_collator=default_data_collator,
)

# === Suppress deprecation warnings ===
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated")

# === Start Training ===
print("=== Starting training === \n")
trainer.train()
print("✅ Training complete.")
