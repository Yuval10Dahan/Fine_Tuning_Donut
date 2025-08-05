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
import gc
import psutil


def print_memory_usage():
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent}% used ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")

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

# Memory-efficient preprocessing function
def preprocess(example):
    try:
        # Load and process image
        image = Image.open(example["image_path"]).convert("RGB")
        
        # Resize image to reduce memory usage (adjust size as needed)
        # Donut typically uses 224x224 or similar
        max_size = 224
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        pixel_values = processor(image, return_tensors="pt").pixel_values[0]
        
        # Close image to free memory
        image.close()
        
        task_prompt = example["task_prompt"]
        target_text = json.dumps(example["ground_truth"], separators=(",", ":"))

        input_ids = processor.tokenizer(
            task_prompt, 
            add_special_tokens=False, 
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).input_ids[0]
        
        labels = processor.tokenizer(
            target_text, 
            add_special_tokens=False, 
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).input_ids[0]
        
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels
        }
    except Exception as e:
        print(f"Error processing {example.get('image_path', 'unknown')}: {e}")
        return None

# Convert to Hugging Face Dataset with memory optimization
print("📦 Processing dataset...")

# Process in smaller batches to avoid memory issues
BATCH_SIZE = 100  # Adjust based on your system memory

def process_dataset_in_batches(data, batch_size=BATCH_SIZE):
    processed_examples = []
    
    for i in range(0, len(data), batch_size):
        print_memory_usage()
        print(f"Processing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        batch = data[i:i + batch_size]
        
        # Process batch
        batch_dataset = Dataset.from_list(batch)
        processed_batch = batch_dataset.map(
            preprocess,
            num_proc=1,  # Use single process to control memory
            remove_columns=batch_dataset.column_names,
            desc=f"Processing batch {i//batch_size + 1}"
        )
        
        # Filter out None values (failed processing)
        processed_batch = processed_batch.filter(lambda x: x is not None)
        
        # Convert to list and extend
        processed_examples.extend(processed_batch.to_list())
        
        # Force garbage collection
        del batch_dataset, processed_batch
        gc.collect()
    
    return Dataset.from_list(processed_examples)

# Process datasets
train_ds = process_dataset_in_batches(train_data)
val_ds = process_dataset_in_batches(val_data)

print(f"Training dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")

# === Custom Data Collator for Memory Efficiency ===
def custom_data_collator(features):
    """Custom data collator that handles memory more efficiently"""
    batch = {}
    
    # Stack pixel values
    batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    
    # Pad input_ids and labels
    max_input_length = max(len(f["input_ids"]) for f in features)
    max_label_length = max(len(f["labels"]) for f in features)
    
    batch["input_ids"] = torch.stack([
        torch.cat([f["input_ids"], torch.full((max_input_length - len(f["input_ids"]),), 
                                            processor.tokenizer.pad_token_id, dtype=torch.long)])
        for f in features
    ])
    
    batch["labels"] = torch.stack([
        torch.cat([f["labels"], torch.full((max_label_length - len(f["labels"]),), 
                                         -100, dtype=torch.long)])
        for f in features
    ])
    
    return batch

# === Training Arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=LOG_DIR,
    save_total_limit=2,
    per_device_train_batch_size=1,  # Keep small for memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Simulate larger batch size
    learning_rate=1e-5,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    dataloader_num_workers=0,  # Disable multiprocessing for stability
    report_to="none",
    logging_steps=50,
    warmup_steps=100,
    max_grad_norm=1.0,
    # Memory optimization
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# === Initialize Trainer ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=processor.tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=custom_data_collator
)

# === Start Training ===
print("🚀 Starting training...")
try:
    trainer.train()
    print("✅ Done training! Model saved to:", OUTPUT_DIR)
except Exception as e:
    print(f"❌ Training failed: {e}")
    # Save checkpoint if training fails
    if hasattr(trainer, 'save_model'):
        trainer.save_model(OUTPUT_DIR + "_checkpoint")
        print(f"💾 Checkpoint saved to: {OUTPUT_DIR}_checkpoint")