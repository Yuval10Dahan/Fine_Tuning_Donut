from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import DonutProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer
from PIL import Image
import torch
import os
import json 

# Set Cache Directory
CACHE_DIR = "./cached_dataset_donut"

# Load Donut Processor
print("Loading Donut Processor...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Step 1: Dataset Preprocessing
def preprocess_data(batch):
    """Processes images and text for Donut model."""
    processed_batch = {"pixel_values": [], "labels": []}

    for image_path, task_prompt, ground_truth in zip(batch["image_path"], batch["task_prompt"], batch["ground_truth"]):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, task_prompt, return_tensors="pt")
        labels = processor.tokenizer(
            json.dumps(ground_truth, separators=(",", ":")),
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).input_ids[0]
        labels[labels == processor.tokenizer.pad_token_id] = -100

        processed_batch["pixel_values"].append(inputs["pixel_values"].squeeze(0))
        processed_batch["labels"].append(labels)

    return processed_batch


if __name__ == "__main__":
    # Step 2: Load or Build Dataset
    if os.path.exists(CACHE_DIR):
        print("Loading Cached Dataset...")
        dataset = DatasetDict.load_from_disk(CACHE_DIR)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        print("Loading Annotations JSON...")
        full_dataset = load_dataset("json", data_files={"train": "annotations.json"})["train"]

        # Optional filtering by categories or any other logic

        # Shuffle and split
        dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        print("Preprocessing Dataset...")
        train_dataset = train_dataset.map(preprocess_data, remove_columns=train_dataset.column_names, batched=True, batch_size=64)
        eval_dataset = eval_dataset.map(preprocess_data, remove_columns=eval_dataset.column_names, batched=True, batch_size=64)

        dataset = DatasetDict({"train": train_dataset, "test": eval_dataset})
        dataset.save_to_disk(CACHE_DIR)
        print("✅ Preprocessed and Saved.")

    # Step 3: Load Donut Model
    print("Loading Donut Model...")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    model.config.decoder.max_length = 512
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    # Step 4: Define Training Arguments
    print("Setting Training Args...")
    training_args = TrainingArguments(
        output_dir="./donut_sensitive_leakage",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_total_limit=2,
        fp16=True,
        logging_dir="./logs"
    )

    # Step 5: Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
    )

    # Step 6: Train
    print("Starting Fine-Tuning...")
    trainer.train()
