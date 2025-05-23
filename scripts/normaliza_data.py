import os
import json
from glob import glob
from datetime import datetime

INPUT_DIR = "data/raw/slack_export"
OUTPUT_DIR = "data/raw/slack_export_normalized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_to_ts(dt_string):
    try:
        dt = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S")
        return f"{dt.timestamp():.6f}".split('.')[0]
    except ValueError:
        return dt_string.split('.')[0]  # si ya es un timestamp válido

for filepath in glob(os.path.join(INPUT_DIR, "*.json")):
    with open(filepath, encoding="utf-8") as f:
        messages = json.load(f)

    for msg in messages:
        if "ts" in msg:
            msg["ts"] = convert_to_ts(msg["ts"])
        if "thread_ts" in msg:
            msg["thread_ts"] = convert_to_ts(msg["thread_ts"])

    output_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

print(f"✅ Archivos normalizados guardados en: {OUTPUT_DIR}")
