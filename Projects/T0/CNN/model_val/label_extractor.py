import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from typing import List
from sklearn.metrics import roc_auc_score

def judger(x, prob_up: List, prob_down: List, per_group: int):
    x = x.tolist()
    max_x = max(x)
    max_idx = x.index(max_x)
    if max_idx == len(x) - 1:
        if max_x >= prob_up[-per_group + 1]:
            return len(x) - 1
    elif max_idx == len(x) - 2:
        if max_x >= prob_up[-per_group]:
            return len(x) - 2

    elif max_idx == 2:
        if max_x >= prob_down[per_group - 1]:
            return 2
    elif max_idx == 1:
        if max_x >= prob_down[per_group - 2]:
            return 1
    #
    # return 0


    # for i in range(1, per_group + 1):
    #     if x[i] > prob_down[i - 1]:
    #         return i
    #     if x[-i] > prob_up[-i]:
    #         return (per_group << 1 | 1) - i
    return 0

def label_extractor(output: torch.Tensor, y_true, cls_num):
    """
    :param sftmx_result: 模型输出的softmax结果
    :param prob_threshold: 某类的概率大于该threshold则为该类，否则为0
    :return: 按threshold提取标签的结果
    """
    assert len(output.shape) == 2
    # assert isinstance(prob_up, list) and isinstance(prob_down, list)

    per_group = cls_num - 1 >> 1

    output: np.array = F.softmax(output, dim = 1).detach().cpu().numpy()

    auc = []
    y_onehot = F.one_hot(y_true.long(), num_classes = 5).detach().numpy()
    for i in range(output.shape[1]):
        # output_i = np.concatenate([output[:, i: i + 1], 1 - output[:, i: i + 1]], axis = 1)
        output_i = output[:, i]
        try:
            auc.append(max(0., min(roc_auc_score(y_onehot[:, i], output_i, average="macro") - 0.1, 1)))
        except ValueError as e:
            auc.append(1)
    # print(auc)
    output_label = np.apply_along_axis(partial(judger, prob_up = auc[-2:],
                                               prob_down = auc[1:3],
                                               per_group = per_group),
                                       axis=1, arr=output)
    output = np.round(output, 3)
    output = np.array([" | ".join(map(str, item)) for item in output])
    return output.reshape((-1, 1)), output_label.reshape((-1, 1))
