import os
import pandas as pd
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")  # e.g., 'C12345678'
SUPPORT_USER_IDS = os.getenv("SUPPORT_USER_IDS", "").split(",")  # Coma separada

client = WebClient(token=SLACK_TOKEN)

def user_is_support(user_id):
    return user_id in SUPPORT_USER_IDS

def fetch_threads(channel_id):
    all_threads = []
    cursor = None
    while True:
        response = client.conversations_history(channel=channel_id, cursor=cursor, limit=200)
        for msg in response["messages"]:
            if msg.get("thread_ts") == msg["ts"]:  # Es hilo
                replies = client.conversations_replies(channel=channel_id, ts=msg["ts"])["messages"]
                all_threads.append(replies)
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return all_threads

def build_qa_dataset(threads):
    qa_pairs = []
    for thread in threads:
        if len(thread) < 2:
            continue

        question_parts = []
        answer = None

        for msg in thread[1:]:
            user = msg.get("user", "")
            text = msg.get("text", "").strip()

            if not text:
                continue

            if answer is None:
                if user_is_support(user):
                    answer = text
                else:
                    question_parts.append(text)
            else:
                break  # Ignoramos después de la primera respuesta

        question = "\n".join(question_parts).strip()

        if question and answer:
            qa_pairs.append((question, answer))

    return qa_pairs

def main():
    print("Descargando mensajes del canal...")
    threads = fetch_threads(CHANNEL_ID)
    print(f"Hilos encontrados: {len(threads)}")

    qa_pairs = build_qa_dataset(threads)
    print(f"Pares pregunta-respuesta extraídos: {len(qa_pairs)}")

    df = pd.DataFrame(qa_pairs, columns=["question", "answer"])
    df.to_csv("data/processed/qa_dataset.csv", index=False)
    print("Dataset guardado en data/processed/qa_dataset.csv")

if __name__ == "__main__":
    main()
