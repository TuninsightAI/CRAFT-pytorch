import base64

import requests


def save_image(image_bytes, image_path="result.jpg"):
    with open(f"{image_path}", "wb") as f:
        f.write(base64.decodebytes(image_bytes.encode('ascii')))


resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('data/屋顶-ALL.png', 'rb')})
save_image(image_bytes=resp.json()["result"], image_path="result.jpg")
