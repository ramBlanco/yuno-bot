import os
import json
import csv
from glob import glob
from collections import defaultdict

INPUT_DIR = "data/raw/slack_export_normalized"  # Carpeta donde están los JSON exportados
OUTPUT_FILE = "data/processed/qa_dataset.csv"

# Almacenar los hilos agrupados por thread_ts
threads = defaultdict(list)

# Recorrer todos los archivos .json
for filepath in glob(os.path.join(INPUT_DIR, "*.json")):
    with open(filepath, encoding="utf-8") as f:
        messages = json.load(f)
        for msg in messages:
            thread_ts = msg.get("thread_ts", msg.get("ts"))
            threads[thread_ts].append(msg)

# Preparar preguntas y respuestas
qa_pairs = []

print(f"Total hilos detectados: {len(threads)}")
print(f"Hilos con al menos 2 mensajes: {sum(1 for m in threads.values() if len(m) > 1)}")

for thread_id, msgs in threads.items():
    sorted_msgs = sorted(msgs, key=lambda m: str(m["ts"]))
    if not sorted_msgs:
        continue

    question = sorted_msgs[0].get("text", "").strip().replace("\n", " ")
    answers = [m.get("text", "").strip().replace("\n", " ") for m in sorted_msgs[1:] if m.get("text")]
    
    if not question or not answers:
        continue

    answer_text = " ".join(answers)
    qa_pairs.append({"question": question, "answer": answer_text})

# Guardar en CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer"])
    writer.writeheader()
    for pair in qa_pairs:
        writer.writerow(pair)

print(f"✅ Dataset generado: {OUTPUT_FILE} ({len(qa_pairs)} pares pregunta-respuesta)")
