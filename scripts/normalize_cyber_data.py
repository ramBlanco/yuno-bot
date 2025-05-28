import pandas as pd

# Leer el archivo CSV
df = pd.read_csv("data/processed/cybersecurity-qa-half.csv")

# Acceder a las preguntas y respuestas
questions = df['question']
answers = df['answer']

# Ejemplo: mostrar las primeras 5 preguntas y respuestas
for q, a in zip(questions.head(), answers.head()):
    print(f"Pregunta: {q}\n")
    print(f"Respuesta: {a}\n")

    print("-"*20)

