import os
import optuna
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
from dotenv import load_dotenv
import torch

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
DATA_PATH = "data/processed/cybersecurity-qa-half.csv"
OUTPUT_DIR = "models/optuna-tuning"
BEST_CONFIG_PATH = "optimize/best_config-v1.json"
MAX_LENGTH = 512

print("model name -------> ", MODEL_NAME)

# === Load and format dataset ===
df = pd.read_csv(DATA_PATH, delimiter=",")
df = df.dropna(subset=["question", "answer"])
print("Dimensions: ")
print(df.shape)

df["prompt"] = df.apply(lambda row: f"Pregunta: {row['question']}\nRespuesta:", axis=1)
hf_dataset = Dataset.from_pandas(df[["prompt", "answer"]])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    inputs = tokenizer(example["prompt"], max_length=MAX_LENGTH, truncation=True, padding="max_length")
    targets = tokenizer(example["answer"], max_length=MAX_LENGTH, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = hf_dataset.map(tokenize, remove_columns=hf_dataset.column_names)
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

# === Subset más pequeño para tuning rápido ===
train_dataset = train_test_split["train"].select(range(min(5000, len(train_test_split["train"]))))
eval_dataset = train_test_split["test"].select(range(min(1000, len(train_test_split["test"]))))

# === Función objetivo para Optuna ===
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [2, 4])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)
    grad_acc_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2])

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"trial-{trial.number}"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=grad_acc_steps,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        evaluation_strategy="no",  # 🔽 no eval por epoch
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, n_jobs=1)

print(f"Best trial:\n  Value: {study.best_trial.value}\n  Params: {study.best_trial.params}")

os.makedirs(os.path.dirname(BEST_CONFIG_PATH), exist_ok=True)
with open(BEST_CONFIG_PATH, "w") as f:
    json.dump(study.best_trial.params, f, indent=4)
