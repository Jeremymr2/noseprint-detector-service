from fastapi import FastAPI
from app.api.predict import model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]
app.add_middleware(
 CORSMiddleware,
 allow_origins=origins,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

@app.get("/")
async def root():
 return {"message": "Model v1"}

app.include_router(model, prefix='/api/v1')