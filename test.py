# test
from data_loader.uci_hand_data import UCIHandPoseDataset
from model.cpm import CPM
from src.util import *

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import ConfigParser

from torch.autograd import Variable
from torch.utils.data import DataLoader


# *********************** hyper parameter  ***********************

# multi-GPU
device_ids = [0, 1, 2, 3]

config = ConfigParser.ConfigParser()
config.read('conf.text')
test_data_dir = config.get('data', 'test_data_dir')
test_label_dir = config.get('data', 'test_label_dir')

learning_rate = config.getfloat('training', 'learning_rate')
batch_size = config.getint('training', 'batch_size')
epochs = config.getint('training', 'epochs')
begin_epoch = config.getint('training', 'begin_epoch')

cuda = torch.cuda.is_available()


model_epo = [10, 15, 20, 25, 30, 35, 40]


def cpm_evaluation(label_map, predict_heatmaps, sigma=0.04):
    """
    calculate the PCK value for one Batch images
    :param label_map:           Batch_size  *  21 *   45  *  45
    :param predict_heatmaps:    Batch_size  *  21 *   45  *  45
    :param sigma:
    :return:
    """
    pck_eval = []
    for b in range(label_map.shape[0]):        # for each batch (person)
        target = np.asarray(label_map[b, :, :, :].data)          # 3D numpy 21 * 45 * 45
        predict = np.asarray(predict_heatmaps[b, :, :, :].data)  # 3D numpy 21 * 45 * 45
        pck_eval.append(PCK(predict, target, sigma=sigma))
    return sum(pck_eval) / float(len(pck_eval))  #


# *************** Build dataset ***************
train_data = UCIHandPoseDataset(data_dir=test_data_dir, label_dir=test_label_dir)
print 'Test dataset total number of images sequence is ----' + str(len(train_data))

# Data Loader
test_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# *************** Build model ***************
net = CPM(out_c=21)


def load_model(model):
    # build model
    net = CPM(out_c=21)
    if torch.cuda.is_available():
        net = net.cuda()
        net = nn.DataParallel(net)  # multi-Gpu

    save_path = os.path.join('ckpt/model_epoch' + str(model)+'.pth')
    state_dict = torch.load(save_path)
    net.load_state_dict(state_dict)
    return net


# **************************************** test all images ****************************************


print '********* test data *********'

for model in model_epo:

    net = load_model(model)
    net.eval()

    pck_dict = {}
    sigma = 0.01
    for i in range(10):
        pck_dict[sigma] = []
        sigma += 0.01

    print 'model epoch ..' + str(model)
    for step, (image, label_map, center_map, imgs) in enumerate(test_dataset):
        image = Variable(image.cuda() if cuda else image)  # 4D Tensor
        # Batch_size  *  3  *  width(368)  *  height(368)

        # 4D Tensor to 5D Tensor
        label_map = torch.stack([label_map] * 6, dim=1)
        # Batch_size  *  41 *   45  *  45
        # Batch_size  *   6 *   41  *  45  *  45
        label_map = Variable(label_map.cuda() if cuda else label_map)  # 5D Tensor

        center_map = Variable(center_map.cuda() if cuda else center_map)  # 4D Tensor
        # Batch_size  *  width(368) * height(368)

        pred_6 = net(image, center_map)  # 5D tensor:  batch size * stages(6) * 41 * 45 * 45

        sigma = 0.01
        for i in range(10):

            # calculate pck
            pck = cpm_evaluation(label_map[:, 5, :, :, :], pred_6[:, 5, :, :, :], sigma=sigma)
            pck_dict[sigma].append(pck)

            if step % 100 == 0:
                print '--step %d ...... sigma %f ...... pck %f' % (step, sigma, pck)
            sigma += 0.01

    print 'Model epoch %d  finished  ==============================>' % (model)

    sigma = 0.01
    results = []
    for i in range(10):
        result = []
        result.append(sigma)
        print 'sigma ==========> ' + str(sigma)
        pck = sum(pck_dict[sigma]) / len(pck_dict[sigma]) * 1.0
        print 'PCK   ==========> ' + str(pck)
        result.append(pck)
        results.append(result)
        sigma += 0.01

    results = pd.DataFrame(results)
    results.to_csv('ckpt/' + str(model) + 'test_pck.csv')



















