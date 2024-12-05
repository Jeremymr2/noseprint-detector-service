from fastapi import FastAPI
from app.api.predict import model

app = FastAPI()

app.include_router(model, prefix='/api/v1/predict')