from http.client import HTTPException

from fastapi import APIRouter

from app.api.model import compare_images

model = APIRouter()

@model.get("/")
async def predict(first_image, second_image):
    try:
        response = compare_images(first_image, second_image)
        return response
    except HTTPException as e:
        raise e