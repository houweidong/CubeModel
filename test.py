import os
import numpy as np
import matplotlib.pyplot as plt
# label_path = '/home/new/dataset/new/labels1/jinshenyi.txt'
# pic_path = '/home/new/dataset/new/pictures/jinshenyi'
#
# picname_list = os.listdir(pic_path)
# with open(label_path) as f:
#     line = f.readline().strip()
#     while line:
#         line = line.split()
#         pic_name = line[0]
#         if line[0] not in picname_list:
#             print(line)
#         line = f.readline().strip()
from torch.nn.functional import linear
import torch.nn as nn
import torch
import numpy as np
# size = 14
# cols = np.arange(0, size, 1, np.float)[np.newaxis, :] + 0.5
# rows = np.arange(0, size, 1, np.float)[:, np.newaxis] + 0.5
# # 1
# yifujinshen_center = [[1 / 2 * size, 1 / 3 * size], [1 / 6 * size, 1 / 2 * size],
#                       [5 / 6 * size, 1 / 2 * size]]
# yifujinshen_sigma = [[1.5 * size / 7, 2.5 * size / 7], [1.5 * size / 7, 3.5 * size / 7],
#                      [1.5 * size / 7, 3.5 * size / 7]]
# yifujinshen_middle = np.exp(
#     - (np.abs((cols - yifujinshen_center[0][0])) ** 3 / (yifujinshen_sigma[0][0] ** 2) +
#        np.abs((rows - yifujinshen_center[0][1])) ** 3 / (
#                (yifujinshen_sigma[0][1] * 2) ** 2)))
# yifujinshen_left = np.exp(
#     - (np.abs((cols - yifujinshen_center[1][0])) ** 3 / (yifujinshen_sigma[1][0] ** 2) +
#        np.abs((rows - yifujinshen_center[1][1])) ** 3 / (yifujinshen_sigma[1][1] ** 2)))
# yifujinshen_right = np.exp(
#     - (np.abs((cols - yifujinshen_center[2][0])) ** 3 / (yifujinshen_sigma[2][0] ** 2) +
#        np.abs((rows - yifujinshen_center[2][1])) ** 3 / (yifujinshen_sigma[2][1] ** 2)))
# yifujinshen_sum = yifujinshen_middle + yifujinshen_left + yifujinshen_right
# yifujinshen = (yifujinshen_sum / np.max(yifujinshen_sum))
#
# # 2
# kuzijinshen_center = [1 / 2 * size, 2 / 3 * size]
# kuzijinshen_sigma = [10 * size / 7, 10 * size / 7]
# kuzijinshen = np.exp(- (np.abs((cols - kuzijinshen_center[0])) ** 4 / (kuzijinshen_sigma[0] ** 2) +
#                         np.abs((rows - kuzijinshen_center[1])) ** 4 / (kuzijinshen_sigma[1] ** 2)))
# kuzijinshen = (kuzijinshen / np.max(kuzijinshen))
#
# # 3
# maozi_center = [[1 / 2 * size, 0 * size], [1 / 2 * size, 1 / 4 * size]]
# maozi_sigma = [3 * size / 7, 1 * size / 7]
# maozi_up = np.exp(- ((np.abs(cols - maozi_center[0][0])) ** 3 / (maozi_sigma[0] ** 2) +
#                      (rows - maozi_center[0][1]) ** 2 / (maozi_sigma[1] ** 2)))
#
# maozi_down = np.exp(- ((np.abs(cols - maozi_center[1][0])) ** 3 / (maozi_sigma[0] ** 2) +
#                        (rows - maozi_center[1][1]) ** 2 / (maozi_sigma[1] ** 2)))
# maozi = maozi_up + maozi_down
# maozi = (maozi / np.max(maozi))
#
# # 4
# gaolingdangbozi_center = [1 / 2 * size, 1 / 6 * size]
# gaolingdangbozi_sigma = [3 * size / 7, 1 * size / 7]
# gaolingdangbozi = np.exp(
#     - ((np.abs(cols - gaolingdangbozi_center[0])) ** 3 / (gaolingdangbozi_sigma[0] ** 2) +
#        (rows - gaolingdangbozi_center[1]) ** 2 / (gaolingdangbozi_sigma[1] ** 2)))
# gaolingdangbozi = (gaolingdangbozi / np.max(gaolingdangbozi))
#
# # 5
# gaofaji_center = [1 / 2 * size, 1 / 24 * size]
# gaofaji_sigma = [3 * size / 7, 1 * size / 7]
# gaofaji = np.exp(- ((np.abs(cols - gaofaji_center[0])) ** 3 / (gaofaji_sigma[0] ** 2) +
#                     (rows - gaofaji_center[1]) ** 2 / (gaofaji_sigma[1] ** 2)))
# gaofaji = (gaofaji / np.max(gaofaji))
#
# plt.matshow(yifujinshen)
# plt.matshow(kuzijinshen)
# plt.matshow(maozi)
# plt.matshow(gaolingdangbozi)
# plt.matshow(gaofaji)
# plt.show()

from model.mobilenetv3 import mobile3l
import torch.nn as nn
from model.model_utils import get_backbone_network, get_param_groups
from model import model_utils
from data.attributes import Attribute
import torch
from utils.opts import parse_opts
from utils.get_tasks import get_tasks
from model.generate_model import generate_model

import torch.optim as optimzer

import numpy as np

# size = 7
#
# cols = np.arange(0, size, 1, np.float)[np.newaxis, :] + 0.5
# rows = np.arange(0, size, 1, np.float)[:, np.newaxis] + 0.5
# #
# attention_center = [1 / 4 * size, 1 / 4 * size]
# attention_sigma = [3 * size / 7, 3 * size / 7]
# attention = np.exp(- (np.abs((cols - attention_center[0])) ** 4 / (attention_sigma[0] ** 2) +
#                         np.abs((rows - attention_center[1])) ** 4 / (attention_sigma[1] ** 2)))
# attention = (attention / np.max(attention))
# #
# # plt.matshow(attention)
# # plt.show()
# # 1
# yifujinshen_center = [[1 / 2 * size, 1 / 3 * size], [1 / 6 * size, 1 / 2 * size],
#                       [5 / 6 * size, 1 / 2 * size]]
# yifujinshen_sigma = [[1.5 * size / 7, 2.5 * size / 7], [1.5 * size / 7, 3.5 * size / 7],
#                      [1.5 * size / 7, 3.5 * size / 7]]
# yifujinshen_middle = np.exp(
#     - (np.abs((cols - yifujinshen_center[0][0])) ** 3 / (yifujinshen_sigma[0][0] ** 2) +
#        np.abs((rows - yifujinshen_center[0][1])) ** 3 / (
#                (yifujinshen_sigma[0][1] * 2) ** 2)))
# yifujinshen_left = np.exp(
#     - (np.abs((cols - yifujinshen_center[1][0])) ** 3 / (yifujinshen_sigma[1][0] ** 2) +
#        np.abs((rows - yifujinshen_center[1][1])) ** 3 / (yifujinshen_sigma[1][1] ** 2)))
# yifujinshen_right = np.exp(
#     - (np.abs((cols - yifujinshen_center[2][0])) ** 3 / (yifujinshen_sigma[2][0] ** 2) +
#        np.abs((rows - yifujinshen_center[2][1])) ** 3 / (yifujinshen_sigma[2][1] ** 2)))
# yifujinshen_sum = yifujinshen_middle + yifujinshen_left + yifujinshen_right
# yifujinshen = (yifujinshen_sum / np.max(yifujinshen_sum))
#
# # 2
# kuzijinshen_center = [1 / 2 * size, 2 / 3 * size]
# kuzijinshen_sigma = [10 * size / 7, 10 * size / 7]
# kuzijinshen = np.exp(- (np.abs((cols - kuzijinshen_center[0])) ** 4 / (kuzijinshen_sigma[0] ** 2) +
#                         np.abs((rows - kuzijinshen_center[1])) ** 4 / (kuzijinshen_sigma[1] ** 2)))
# kuzijinshen = (kuzijinshen / np.max(kuzijinshen))
#
# # 3
# maozi_center = [[1 / 2 * size, 0 * size], [1 / 2 * size, 1 / 4 * size]]
# maozi_sigma = [3 * size / 7, 1 * size / 7]
# maozi_up = np.exp(- ((np.abs(cols - maozi_center[0][0])) ** 3 / (maozi_sigma[0] ** 2) +
#                      (rows - maozi_center[0][1]) ** 2 / (maozi_sigma[1] ** 2)))
#
# maozi_down = np.exp(- ((np.abs(cols - maozi_center[1][0])) ** 3 / (maozi_sigma[0] ** 2) +
#                        (rows - maozi_center[1][1]) ** 2 / (maozi_sigma[1] ** 2)))
# maozi = maozi_up + maozi_down
# maozi = (maozi / np.max(maozi))
#
# # 4
# gaolingdangbozi_center = [1 / 2 * size, 1 / 6 * size]
# gaolingdangbozi_sigma = [3 * size / 7, 1 * size / 7]
# gaolingdangbozi = np.exp(
#     - ((np.abs(cols - gaolingdangbozi_center[0])) ** 3 / (gaolingdangbozi_sigma[0] ** 2) +
#        (rows - gaolingdangbozi_center[1]) ** 2 / (gaolingdangbozi_sigma[1] ** 2)))
# gaolingdangbozi = (gaolingdangbozi / np.max(gaolingdangbozi))
#
# # 5
# gaofaji_center = [1 / 2 * size, 1 / 24 * size]
# gaofaji_sigma = [3 * size / 7, 1 * size / 7]
# gaofaji = np.exp(- ((np.abs(cols - gaofaji_center[0])) ** 3 / (gaofaji_sigma[0] ** 2) +
#                     (rows - gaofaji_center[1]) ** 2 / (gaofaji_sigma[1] ** 2)))
# gaofaji = (gaofaji / np.max(gaofaji))
#
# plt.matshow(yifujinshen)
# plt.matshow(kuzijinshen)
# plt.matshow(maozi)
# plt.matshow(gaolingdangbozi)
# plt.matshow(gaofaji)
# plt.matshow(attention)
# plt.show()

rec_available = 0
mask = []
mask.append(torch.tensor([rec_available], dtype=torch.bool, requires_grad=False))
print(mask)