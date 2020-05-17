"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import argparse
import base64
import io
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from flask import Flask, request, jsonify
from torch.autograd import Variable

import craft_utils
import file_utils
import imgproc
from craft import CRAFT

app = Flask(__name__)


def img_iter(image: np.ndarray, step: int):
    x, y = image.shape[:2]
    xs = np.arange(0, x - step, step)
    xs = np.append(xs, x - step)
    ys = np.arange(0, y - step, step)
    ys = np.append(ys, y - step)
    for _x in xs:
        for _y in ys:
            yield image[_x:_x + step, _y:_y + step, :], [[_x, _x + step], [_y, _y + step]]


def get_image_bytes(image_array: np.ndarray):
    img = Image.fromarray(image_array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img


def randomString(stringLength=8):
    import random
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        result_encoding = test_single_image(net, img_bytes, args.text_threshold, args.link_threshold,
                                            args.low_text,
                                            args.cuda, args.poly)
        return jsonify({"message": "okay", "result": result_encoding})
    else:
        return "hello"


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def binary_image(RGB_image: np.ndarray):
    image = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2GRAY)
    image = np.repeat(image[:, :, None], [3], axis=2)
    image = ((image > 240) * 255).astype(np.uint8)
    return image


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.2, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.1, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.2, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=500, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')

args = parser.parse_args()


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_AREA,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()
    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    return boxes, polys, ret_score_text


def test_single_image(net, image_bytes, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.asarray(image)
    result_image = np.zeros_like(image)
    image_path_gen = img_iter(image, args.canvas_size)
    for image_patch, coord in image_path_gen:
        result_patch = _test_patch_image(net, image_patch, text_threshold, link_threshold, low_text, cuda, poly)
        result_image[coord[0][0]:coord[0][1], coord[1][0]:coord[1][1]] = result_patch
    result_encoding = get_image_bytes(result_image)
    return result_encoding


def _test_patch_image(net, image_patch, text_threshold, link_threshold, low_text, cuda, poly):
    bboxes, polys, score_text = test_net(net, image_patch, text_threshold, link_threshold, low_text,
                                         cuda, poly)
    random_name = randomString(8)
    result_patch = file_utils.saveResult(random_name, image_patch[:, :, ::-1], polys, dirname="api_result/",
                                         return_matrix=True)
    return result_patch


if __name__ == '__main__':
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    app.run(host="0.0.0.0")
