import base64

import numpy as np
import requests
from PIL import Image


def bbox_iter(bbox):
    for (cordinate, _bbox) in bbox:
        for b in _bbox:
            yield (int(cordinate[1][0] + b[0][0]), int(cordinate[1][0] + b[1][0]),
                   int(cordinate[0][0] + b[1][1]), int(cordinate[0][0] + b[2][1], ))


def save_image(image_bytes, image_path="result.jpg"):
    with open(f"{image_path}", "wb") as f:
        f.write(base64.decodebytes(image_bytes.encode('ascii')))
    return np.array(Image.open(f"{image_path}").convert("RGB"))


def get_image_crop(image, b):
    return image[b[2]:b[3], b[0]:b[1]].copy()


resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('data/屋顶-ALL.png', 'rb'),
                            "draw_contour": False})
image = save_image(image_bytes=resp.json()["result"], image_path="result_300.jpg")
bbox = eval(resp.json()["bbox"])
bbox = bbox_iter(bbox)
croped_images = []
croped_coordinate = []

for b in bbox:
    croped_images.append(get_image_crop(image, b))
    croped_coordinate.append(b)
print(f"Found {len(croped_images)} croped images.")
import matplotlib.pyplot as plt

for p in croped_images:
    plt.clf()
    plt.imshow(p)
    plt.show(block=False)
    plt.pause(0.1)
