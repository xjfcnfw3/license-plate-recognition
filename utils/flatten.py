import cv2
import os
from IPython.display import Image, clear_output
import numpy as np
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

# 추후 모듈 통합에 의해 수정될 수 있음
DATA_PATH = './train/_annotations.coco.json'
f = open(DATA_PATH)
f = f.read()
json_object = json.loads(f)

PATH = "./train/"


def make_files_metadata(json_object):
    img_data = []
    for data in json_object["images"]:
        img_data.append(data)
    return img_data

def make_points(json_object):
    points_list = []
    for data in json_object['annotations']:
        arr = data['segmentation'][0]
        points = [[arr[2], arr[3]], [arr[0], arr[1]], [arr[4], arr[5]], [arr[6], arr[7]]]
        points_list.append(points)
    return points_list


def show_data(json_object):
    img_data = make_files_metadata(json_object)
    points_list = make_points(json_object)
    for i in range(len(img_data)):
    # for i in range(1):
        # for i in range(1):
        img = cv2.imread(PATH + img_data[i]['file_name'])
        cols, row = img_data[i]['width'], img_data[i]['height']
        points = points_list[i]
        box = [[0, 0], [cols, 0], [0, row], [cols, row]]

        pts1 = np.float32(points)
        pts2 = np.float32(box)
        points = np.int32(points)

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (cols, row))  # 변환후 크기 (x좌표, y좌표)

        # cv2.imshow("data", dst)
        # cv2.waitKey(0)
        # break;
        cv2.imwrite('./data/' + img_data[i]['file_name'].split('_jpg')[0] + '.jpg', dst)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./data/' + img_data[i]['file_name'].split('_jpg')[0] + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
show_data(json_object)