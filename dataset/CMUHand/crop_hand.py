"""
Process CMU Hand dataset to get cropped hand datasets.

"""
import os
import numpy as np
import json

from PIL import Image


dirs = ['train', 'test']

for data in dirs:

    savd_dir = 'hand_labels/' + data + '/crop/'
    new_label_dir = 'hand_labels/' + data + '/crop_label/'

    imgs = os.listdir('hand_labels/' + data + '/data/')
    for img in imgs:
        if img == '.DS_Store':
            continue

        data_dir = 'hand_labels/' + data + '/data/' + img
        label_dir = 'hand_labels/' + data + '/label/' + img[:-4] + '.json'

        dat = json.load(open(label_dir))
        pts = np.array(dat['hand_pts'])

        xmin = min(pts[:, 0])
        xmax = max(pts[:, 0])
        ymin = min(pts[:, 1])
        ymax = max(pts[:, 1])

        B = max(xmax - xmin, ymax - ymin)
        # B is the maximum dimension of the tightest bounding box
        width = 2.2 * B     # This is based on the paper

        # the center of hand box can be
        center = dat["hand_box_center"]
        hand_box = [[center[0] - width / 2., center[1] - width / 2.],
                    [center[0] + width / 2., center[1] + width / 2.]]
        hand_box = np.array(hand_box)

        im = Image.open(data_dir)
        im = im.crop((hand_box[0, 0], hand_box[0, 1], hand_box[1, 0], hand_box[1, 1]))
        im = im.resize((368, 368))

        im.save(savd_dir + img)  # save cropped image

        lbl = pts[:, :2] - hand_box[0, :]
        lbl = lbl * 368 / width
        lbl = lbl.tolist()

        label_dict = {}
        label_dict['hand_pts_crop'] = lbl
        json.dump(label_dict, open(new_label_dir + img[:-4] + '.json', 'w'))






