import cv2
import ray
import requests
import torch
import numpy as np
from util import draw_bbox_array, make_csv, process_image, non_max_suppression


def server_request(d):
    url = "http://165.246.121.112:5001/invocations"
    H = {"Content-Type": "application/json"}
    print(len(d))
    D = {"inputs": d}
    res = requests.post(url=url, json=D, headers=H)
    if res.status_code == 200:
        result = res.json()
        return result["predictions"]
    else:
        print("Request failed with status code:", res.status_code)
        print(f"error : {res.text}")


def Infer(img, conf_thres, iou_thres):
    if isinstance(img, np.ndarray) and len(img.shape) == 3:
        img = img[None]
    img = torch.stack([process_image(i, (640, 640), stride=32, half=False)[0] for i in img])
    pred = server_request(np.array(img).tolist())
    det = non_max_suppression(
        torch.tensor(pred),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=None,
        agnostic=False,
        max_det=1000,
    )
    return det


def online_infer(image, conf_thres, iou_thres, img_shape=(640, 640), sic=False, only_det=False):
    """
    only_det = False 인 경우 : 한번에 한장의 이미지를 infer하는 경우 - image : img 파일의 주소 str
    only_det = True  인 경우 : 한번에 여러장 이미지를 infer하는 경우 - image : img numpy array
    """

    if only_det:
        image_name = image
        image = cv2.imread(image)
        det = Infer(image, conf_thres, iou_thres)
        det = draw_bbox_array(det, img_shape, image, sic, only_det)
        csvs = make_csv(det)[0]
        csv_names = ".".join("/".join(image_name.split("/")[1:]).split(".")[:-1] + ["csv"])
        return csvs, csv_names
    else:
        det = Infer(image, conf_thres, iou_thres)
        draw_img_array, det = draw_bbox_array(det[0], (640, 640), image, sic, only_det)
        csv = make_csv(det)[0]
        return draw_img_array, csv


@ray.remote
def batch_infer(img_path_list, conf_thres, iou_thres, img_shape=(640, 640), data_path=""):
    if not isinstance(img_path_list, list):
        img_path_list = [img_path_list]
    image = [cv2.imread(f) for f in img_path_list]
    det = Infer(image, conf_thres, iou_thres)
    det = draw_bbox_array(det, img_shape, image, sic=False, only_det=True)
    csvs = make_csv(det)

    csv_names = [".".join(file.split(".")[:-1] + ["csv"])[len(data_path) :] for file in img_path_list]
    return csvs, csv_names


if __name__ == "__main__":
    pass
