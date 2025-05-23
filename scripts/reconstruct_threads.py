import os
import json
from glob import glob
from datetime import datetime
from collections import defaultdict

INPUT_DIR = "data/raw/slack_export"
OUTPUT_FILE = "data/processed/qa_dataset_reconstructed.csv"

# Cargar todos los mensajes en orden cronológico
all_messages = []
for filepath in glob(os.path.join(INPUT_DIR, "*.json")):
    with open(filepath, encoding="utf-8") as f:
        messages = json.load(f)
        for msg in messages:
            if "ts" not in msg:
                continue
            try:
                msg_ts = datetime.strptime(msg["ts"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                msg_ts = datetime.fromtimestamp(float(msg["ts"]))
            msg["parsed_ts"] = msg_ts
            all_messages.append(msg)

# Ordenar mensajes por timestamp
all_messages.sort(key=lambda m: m["parsed_ts"])

# Reconstruir hilos: asumimos que un hilo comienza con un mensaje y las respuestas son las siguientes dentro de un tiempo límite
threads = []
current_thread = []
last_msg_time = None
THREAD_TIMEOUT_MINUTES = 30

for msg in all_messages:
    if not current_thread:
        current_thread.append(msg)
        last_msg_time = msg["parsed_ts"]
    else:
        time_diff = (msg["parsed_ts"] - last_msg_time).total_seconds() / 60
        if time_diff <= THREAD_TIMEOUT_MINUTES:
            current_thread.append(msg)
            last_msg_time = msg["parsed_ts"]
        else:
            if len(current_thread) > 1:
                threads.append(current_thread)
            current_thread = [msg]
            last_msg_time = msg["parsed_ts"]

# Agregar el último hilo si es válido
if len(current_thread) > 1:
    threads.append(current_thread)

# Crear pares pregunta-respuesta
qa_pairs = []
for thread in threads:
    question = thread[0].get("text", "").strip().replace("\n", " ")
    answers = [m.get("text", "").strip().replace("\n", " ") for m in thread[1:] if m.get("text")]
    if question and answers:
        qa_pairs.append({"question": question, "answer": " ".join(answers)})

# Guardar CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
import csv
with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer"])
    writer.writeheader()
    for pair in qa_pairs:
        writer.writerow(pair)

print(f"✅ Dataset reconstruido con {len(qa_pairs)} pares pregunta-respuesta")