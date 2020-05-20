import base64

import numpy as np
import requests
from PIL import Image


def ocr_space_file(filename, overlay=False, api_key='dfbff62ed388957', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()


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
                     files={"file": open('data/9-ALL.png', 'rb'),
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
# res = ocr_space_file("result_300.jpg")

import matplotlib.pyplot as plt
import json

margin = 0
for p in croped_images:
    fp = "tmp.png"
    Image.fromarray(np.pad(p, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)).save(fp)
    res = json.loads(ocr_space_file(fp, language="chs"))
    print(res["ParsedResults"][0]["ParsedText"])
    plt.clf()
    plt.imshow(p)
    plt.show(block=False)
    plt.pause(0.1)
