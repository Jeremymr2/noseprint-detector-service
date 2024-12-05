from http.client import HTTPException

from fastapi import Depends
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import cv2


# load model
def load_model():
    x1 = layers.Input(shape=(96, 96, 1))
    x2 = layers.Input(shape=(96, 96, 1))

    # share weights both inputs
    inputs = layers.Input(shape=(96, 96, 1))

    feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    feature = layers.MaxPooling2D(pool_size=2)(feature)

    feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(feature)
    feature = layers.MaxPooling2D(pool_size=2)(feature)

    feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(feature)
    feature = layers.MaxPooling2D(pool_size=2)(feature)

    feature_model = Model(inputs=inputs, outputs=feature)
    # 2 feature models that sharing weights
    x1_net = feature_model(x1)
    x2_net = feature_model(x2)

    # subtract features
    net = layers.Subtract()([x1_net, x2_net])

    net = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(net)
    net = layers.MaxPooling2D(pool_size=2)(net)

    net = layers.Flatten()(net)

    net = layers.Dense(64, activation='relu')(net)

    net = layers.Dense(1, activation='sigmoid')(net)

    model = Model(inputs=[x1, x2], outputs=net)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    model.load_weights("./app/models/siamese.h5")

    return model

def unsharpMasking(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    mask = gray - blurred
    alpha = 2
    unsharp = gray + alpha * mask
    return unsharp

def preprocess_numpy(img):
    img = unsharpMasking(img)

    img = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

    img = cv2.resize(img, (96, 96))
    return img

def reshape_numpy(img):
    img = img.reshape((1, 96, 96, 1)).astype(np.float32) / 255
    return img

# main function
def compare_images(first_img: np.ndarray, second_img: np.ndarray, model = Depends(load_model)):
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found")
    # if first_img is a file
    # np_array = np.frombuffer(first_img, dtype=np.uint8)
    # img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    img_numpy = preprocess_numpy(first_img)
    img_numpy = reshape_numpy(img_numpy)
    # if second_img is a file
    # np_url = np.frombuffer(response.content, dtype=np.uint8)
    # img_url = cv2.imdecode(np_url, cv2.IMREAD_COLOR)
    img_url = preprocess_numpy(second_img)
    img_url = reshape_numpy(img_url)
    result = model.predict([img_numpy, img_url])
    return result