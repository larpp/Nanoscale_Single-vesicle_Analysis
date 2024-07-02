# 2024.06.28, Add Infernece file

# You can check the detail here
# https://github.com/larpp/MMDetection_Inference

from mmdet.apis import DetInferencer
import glob
import csv
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection Inference settings')

    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--conf_thr', type=int, default=0.5, help='confidence score threshold')
    parser.add_argument('--inference_path', type=str, default='RESULTS_final', help='file path with images to Inference')
    parser.add_argument('--out_dir', type=str, default='FINAL_Faster_RCNN_results', help='Inference results directory')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint
    pred_score_thr = args.conf_thr

    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device='cuda')

    # Inference할 이미지들이 있는 경로
    image_li = glob.glob(f'{args.inference_path}/*.png')
    results_path = args.out_dir
    os.makedirs(f'{results_path}/data_csv') # Predict 값을 저장할 디렉터리 생성

    for img in image_li:
        image_name = img.split('/')[-1][:-4]

        outputs = inferencer(img, out_dir=results_path, pred_score_thr=pred_score_thr)
        scores = outputs['predictions'][0]['scores']
        boxes = outputs['predictions'][0]['bboxes']
        labels = outputs['predictions'][0]['labels']


        columns = ['labels', 'x_center', 'y_center', 'width', 'height', 'confidence_score', 'area']
        info = []
        for idx, box in enumerate(boxes):
            confidence = scores[idx]
            if confidence < pred_score_thr:
                break

            w, h = box[2] - box[0], box[3] - box[1]
            x_c = box[0] + w/2
            y_c = box[1] + h/2
            area = w * h
            data = [labels[idx], x_c, y_c, w, h, confidence, area]
            info.append(data)


        with open(f'{results_path}/data_csv/{image_name}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(columns)

            for value in info:
                writer.writerow(value)


if __name__=='__main__':
    main()
