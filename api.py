from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    # Aquí iría tu lógica con el modelo cargado
    return {"answer": "respuesta de ejemplo"}
