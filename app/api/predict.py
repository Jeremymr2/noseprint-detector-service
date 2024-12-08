from http.client import HTTPException
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.api.model import compare_images, load_model
from fastapi import UploadFile
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import base64

model = APIRouter()

class ImageRequest(BaseModel):
    image_1: str
    image_2: str

@model.post("/predict")
async def predict(images: ImageRequest, comparator = Depends(load_model)):
    # Decodificar ambas imágenes base64
    image_data_1 = base64.b64decode(images.image_1)
    image_data_2 = base64.b64decode(images.image_2)

    response = compare_images(image_data_1, image_data_2, comparator)

    # try:
    #     response = compare_images(first_image, second_image, comparator)
    #     return response
    # except HTTPException as e:
    #     raise e
    return response
