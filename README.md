# Folders

```
root/
│
├── data/                     # Datos crudos y procesados
│   ├── raw/                  # Mensajes originales de Slack
│   └── processed/            # Dataset limpio QA (CSV o parquet)
│
├── models/                   # Checkpoints del modelo entrenado
│   └── fine-tuned/           # Modelo ya entrenado con nuestros datos
│
├── notebooks/                # Exploración, visualización y pruebas en Jupyter
│
├── scripts/                  # Scripts ejecutables
│   ├── fetch_slack_data.py   # Descarga mensajes desde Slack
│   ├── preprocess_data.py    # Limpieza y creación del dataset QA
│   ├── train.py              # Entrenamiento/fine-tuning del modelo
│   ├── optimize.py           # Optuna para búsqueda de hiperparámetros
│   └── run_inference.py      # Pruebas con el modelo (predicción)
│
├── slack_bot/                # (Opcional) Código para integrar con Slack como bot
│   ├── app.py                # FastAPI o Flask
│   └── bot_utils.py          # Helpers de Slack
│
├── config/                   # Configs para modelos, entrenamiento, Slack, etc.
│   ├── training_config.yaml
│   └── slack_config.yaml
│
├── requirements.txt          # Librerías del entorno
├── README.md
└── .env                      # Token de Slack y claves (usado por dotenv)
```