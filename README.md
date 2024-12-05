# Noseprint Detector Service

This API, built with FastAPI, enables image comparison for facial and nasal pattern recognition, returning results based on a pre-trained model.

## Features
- Image Comparison: Compares two images to determine similarity using a machine learning model.
- Error Handling: Manages HTTP errors if the model is unavailable or there are issues with the provided images.


## Prerequisites

- Python 3.11
- Virtual environment (optional)
- Install dependencies (requirements.txt)

## Installation
1. Clone the repository
```sh
git clone https://github.com/Jeremymr2/noseprint-detector-service.git
cd noseprint-detector-api
```
2. Create and activate a virtual environment
```sh
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```
3. Install dependencies
```sh
pip install -r requirements.txt
```

## Usage

1. Run the api
```sh
uvicorn main:app --reload
```
2. Available endpoints
- GET /api/v1/predict/
  - Description: Compares two images provided as input and returns the similarity score
- Input parameters:
  - `first_image` (image file in NumPy format)
  - `second_image` (image file in NumPy format)
- Example response:
```sh
{
    "result": [[0.87]]"
}
```
- Possible Errors:
  - `404 Model not found` The model is not available

## Project Structure
```
noseprint-detector-api/
│
├── app/
│   ├── api/                  # Contains API-related logic
│   ├── main.py               # Main API code
│   └── models/               # Model .h5
│
├── .venv/                    # Virtual environment (optional)
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```