import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from dotenv import load_dotenv
import torch
import json
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

load_dotenv()

# === CONFIG ===
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
DATA_PATH = "data/processed/qa_dataset_reconstructed.csv"
OUTPUT_DIR = "models/fine-tuned"
MAX_LENGTH = 512

with open("optimize/best_config.json", "r") as f:
    best_params = json.load(f)

# === Load and preprocess dataset ===
df = pd.read_csv(DATA_PATH)

def format_prompt(row):
    return f"Pregunta: {row['question']}\nRespuesta:"  # estilo Instruct

df["prompt"] = df.apply(format_prompt, axis=1)

hf_dataset = Dataset.from_pandas(df[["prompt", "answer"]])

print("model name -------> ", MODEL_NAME)

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    inputs = tokenizer(
        example["prompt"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )
    targets = tokenizer(
        example["answer"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

print("➡️ Tokenizing dataset...")
tokenized_dataset = hf_dataset.map(tokenize, remove_columns=hf_dataset.column_names)
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print("✅ Tokenization done")

# === Model ===
print("➡️ Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("✅ Model loaded")

# === Training config ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=best_params.get("per_device_train_batch_size", 4),
    per_device_eval_batch_size=best_params.get("per_device_train_batch_size", 8),
    learning_rate=best_params.get("learning_rate", 5e-5),
    num_train_epochs=best_params.get("num_train_epochs", 3),

    gradient_accumulation_steps=8,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    evaluation_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    report_to="none",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# === Save final model ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModelo guardado en: {OUTPUT_DIR}")
