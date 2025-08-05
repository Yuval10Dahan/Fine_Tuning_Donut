from datasets import load_dataset, DatasetDict
from transformers import DonutProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import LoraModel
from peft.utils import find_all_linear_names
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

        # Reduce dataset size for debugging
        full_dataset = full_dataset.select(range(500))  # Try with just 500 examples

        dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        print("Preprocessing Dataset...")
        train_dataset = train_dataset.map(preprocess_data, remove_columns=train_dataset.column_names, batched=True, batch_size=32)
        eval_dataset = eval_dataset.map(preprocess_data, remove_columns=eval_dataset.column_names, batched=True, batch_size=32)

        dataset = DatasetDict({"train": train_dataset, "test": eval_dataset})
        dataset.save_to_disk(CACHE_DIR)
        print("✅ Preprocessed and Saved.")

    # Step 3: Load Donut Model
    print("Loading Donut Model...")
    base_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    base_model.config.decoder.max_length = 512
    base_model.config.eos_token_id = processor.tokenizer.eos_token_id

    # Step 4: Apply LoRA
    print("Applying LoRA...")
    target_modules = find_all_linear_names(base_model)
    print("✅ LoRA will be applied to:", target_modules)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(base_model, peft_config)

    # Step 5: Define Training Arguments
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

    # Step 6: Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
    )

    # Step 7: Train
    print("Starting Fine-Tuning...")
    trainer.train()
