import json
import random
import cv2
from cv2 import imshow

f = open("test.json")
j = json.load(f)
f.close()

images = j['images']
img_mapping = {}

for img_info in images:
    img_mapping[img_info['id']] = img_info['file_name']

annotations = j['annotations']

for i, anno in enumerate(annotations):
    img = cv2.imread(img_mapping[anno['image_id']])
    # print(anno['bbox'])
    x, y, w, h = anno['bbox']
    ih, iw, _ = img.shape
    # delta width for padding
    dw = w * 0.1
    dh = h * 0.1
    x = max(x - dw, 0)
    y = max(y - dh, 0)

    w = min(x + w + dw * 2, iw) - x
    h = min(y + h + dh * 2, ih) - y
    if float(w)/float(h) < 1.75:
        continue
    to_save = img[int(y):int(y+h), int(x):int(x+w)]
    cv2.imwrite("test/img_" + str(i) + "-" + str(i) + '.jpg', img)

    # clean that rect area, so not to train it as neg dataset
    # img[int(y):int(y+h), int(x):int(x+w)] = 0

    # for j, itr in enumerate(range(100)):
    #     nx = random.randint(0, int(iw))
    #     ny = random.randint(0, int(ih))
    #     nw = random.randint(10, max(10, int(iw - nx)))
    #     nh = random.randint(10, max(10, int(ih - ny)))

    #     if min(nx + nw, iw) == iw:
    #         continue

    #     if min(ny + nh, ih) == ih:
    #         continue
    #     print(int(ny), int(ny+nh), int(nx), int(nx+nw))
    #     cv2.imwrite("neg/neg_img_" + str(i) + "-" + str(j) +
    #                 '.jpg', img[int(ny):int(ny+nh), int(nx):int(nx+nw)])

    # cv2.waitKey(0)
    # cv2.imwrite("neg/neg_img_" + str(i) + "-" + str(i) + '.jpg', img)
print(img_mapping)
