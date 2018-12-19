"""
save pck values and label json file for CPM
step 1: predict label and save into json file for every image
step 2: save pck value for all images
"""

from handpose_data_cpm import UCIHandPoseDataset
from cpm import CPM

import ConfigParser
import pandas as pd
import numpy as np
import os
import scipy.misc
import json

import torch
import torch.nn as nn
from torch.autograd import Variable


from torch.utils.data import DataLoader


# *********************** hyper parameter  ***********************


device_ids = [0, 1, 2, 3]        # multi-GPU

config = ConfigParser.ConfigParser()
config.read('conf.text')
test_data_dir = config.get('data', 'test_data_dir')
test_label_dir = config.get('data', 'test_label_dir')
save_dir = config.get('data', 'save_dir')


batch_size = config.getint('training', 'batch_size')

best_model = config.getint('test', 'best_model')
predict_label_dir = config.get('test', 'predict_label_dir')
predict_labels_dir = config.get('test', 'predict_labels_dir')

pck_save_dir = config.get('test', 'pck_save_dir')

cuda = torch.cuda.is_available()

sigma = 0.04


# *********************** function ***********************
if not os.path.exists(predict_label_dir):
    os.mkdir(predict_label_dir)

if not os.path.exists(predict_labels_dir):
    os.mkdir(predict_labels_dir)


def PCK(predict, target, label_size=45, sigma=0.04):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         3D numpy       21 * 45 * 45
    :param target:          3D numpy       21 * 45 * 45
    :param label_size:
    :param sigma:
    :return: 0/21, 1/21, ...
    """
    pck = 0
    for i in range(predict.shape[0]):       # 21

        pre_x, pre_y = np.where(predict[i, :, :] == np.max(predict[i, :, :]))
        tar_x, tar_y = np.where(target[i, :, :] == np.max(target[i, :, :]))

        dis = np.sqrt((pre_x[0] - tar_x[0])**2 + (pre_y[0] - tar_y[0])**2)
        if dis < sigma * label_size:
            pck += 1

    return pck / 21.0


def Tests_save_label_PCK(label_map, predict_heatmaps, step, imgs):
    """
    :param label_map:           4D Tensor    batch size * 41 * 45 * 45
    :param predict_heatmaps:    4D Tensor    batch size * 41 * 45 * 45
    :param step:
    :param imgs:                batch_size * 1
    :return:
    """
    pck_dict = {}
    for b in range(label_map.shape[0]):  # for each batch (person)
        seq = imgs[b].split('/')[-2]  # sequence name 001L0
        label_dict = {}  # all image label in the same seq

        labels_list = []  # 21 points label for one image [[], [], [], .. ,[]]
        im = imgs[b].split('/')[-1][1:5]  # image name 0005

        # ****************** get pck of one image ************************
        target = np.asarray(label_map[b, :, :, :])               # 3D numpy 21 * 45 * 45
        predict = np.asarray(predict_heatmaps[b, :, :, :].data)  # 3D numpy 21 * 45 * 45

        pck = PCK(predict, target, sigma=0.04)
        pck_dict[seq + '_' + im] = pck

        # ****************** save image and label of 21 joints ******************
        for i in range(21):  # for each joint
            tmp_pre = np.asarray(predict_heatmaps[b, i, :, :].data)  # 2D
            #  get label of original image
            corr = np.where(tmp_pre == np.max(tmp_pre))
            x = corr[0][0] * (256.0 / 45.0)
            x = int(x)
            y = corr[1][0] * (256.0 / 45.0)
            y = int(y)
            labels_list.append([y, x])  # save img label to json

        label_dict[im] = labels_list  # save label

        # calculate average PCK
        avg_pck = sum(pck_dict.values()) / float(pck_dict.__len__())
        print 'step ...%d ... PCK %f  ....' % (step, avg_pck)

        # ****************** save label ******************
        save_dir_label = predict_label_dir + '/' + seq          # 101L0
        if not os.path.exists(save_dir_label):
            os.mkdir(save_dir_label)
        json.dump(label_dict, open(save_dir_label + '/' + str(step) + '.json', 'w'), sort_keys=True, indent=4)
    return pck_dict


# ************************************ Build dataset ************************************
test_data = UCIHandPoseDataset(data_dir=test_data_dir, label_dir=test_label_dir)
print 'Test dataset total number of images sequence is ----' + str(len(test_data))

# Data Loader
test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Build model
net = CPM(21)
if cuda:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)  # multi-Gpu

model_path = os.path.join('ckpt/model_epoch' + str(best_model)+'.pth')
state_dict = torch.load(model_path)
net.load_state_dict(state_dict)


# **************************************** test all images ****************************************

print '********* test data *********'
net.eval()

all_pcks = {}  # {0005:[[], [],[]], 0011:[[], [],[]] }

for step, (image, label_map, center_map, imgs) in enumerate(test_dataset):
    image = Variable(image.cuda() if cuda else image)   # 4D Tensor
    # Batch_size  *  3  *  width(368)  *  height(368)
    label_map = torch.stack([label_map] * 6, dim=1)     # 4D Tensor to 5D Tensor
    # Batch_size  *   6 *   41  *  45  *  45
    label_map = Variable(label_map.cuda() if cuda else label_map)  # 5D Tensor

    center_map = Variable(center_map.cuda() if cuda else center_map)  # 4D Tensor
    # Batch_size  *  width(368) * height(368)

    pred_6 = net(image, center_map)  # 5D tensor:  batch size * stages(6) * 41 * 45 * 45
    # calculate pck

    # ****************** calculate pck  ******************
    pck_dict = Tests_save_label_PCK(label_map[:, 5, :, :, :], pred_6[:, 5, :, :, :], step, imgs=imgs)
    all_pcks = dict(pck_dict.items() + all_pcks.items())


print '===PCK evaluation in test dataset is ' + str(sum(all_pcks.values()) / all_pcks.__len__())


df = pd.DataFrame(list(all_pcks.items()), columns=['img', 'pck'])
df = df.sort_values(by='img', ascending=True)           # sort by name
df.to_csv(pck_save_dir)


# ****************** merge label json file ******************

print 'merge json file ............ '

seqs = os.listdir(predict_label_dir)

for seq in seqs:
    if seq == '.DS_Store':
        continue
    print seq

    s = os.path.join(predict_label_dir, seq)
    steps = os.listdir(s)
    d = {}
    for step in steps:
        lbl = json.load(open(s + '/' + step))
        d = dict(d.items() + lbl.items())

    json.dump(d, open(predict_labels_dir + '/' + seq + '.json', 'w'), sort_keys=True, indent=4)

os.system('rm -r '+predict_label_dir)

