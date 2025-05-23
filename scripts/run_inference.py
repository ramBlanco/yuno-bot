import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
MODEL_DIR = os.getenv("MODEL_DIR", "models/fine-tuned")
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 128

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

def answer_question(question: str) -> str:
    prompt = f"Pregunta: {question}\nRespuesta:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_INPUT_LENGTH
    )

    output = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=MAX_OUTPUT_LENGTH,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    print("\U0001F4AC Ingreso de preguntas (Ctrl+C para salir):\n")
    try:
        while True:
            question = input(">> ")
            response = answer_question(question)
            print("Respuesta:", response)
    except KeyboardInterrupt:
        print("\nSaliendo.")
