import csv
import json
import os

CSV_INPUT_PATH = "data/raw/messages.csv"
JSON_OUTPUT_PATH = "data/raw/messages.json"

# Leer el CSV
with open(CSV_INPUT_PATH, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    data = [row for row in reader]

# Guardar como JSON
with open(JSON_OUTPUT_PATH, mode="w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=2, ensure_ascii=False)

print(f"✅ Archivo convertido: {JSON_OUTPUT_PATH} ({len(data)} registros)")
