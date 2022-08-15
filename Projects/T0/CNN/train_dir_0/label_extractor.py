import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from functools import partial

def judger(x, prob_threshold):
    if x[0] > prob_threshold:
        return 1
    elif x[1] > prob_threshold:
        return 2
    else:
        return 0

def label_extractor(output: torch.Tensor, prob_threshold: float):
    """
    :param sftmx_result: 模型输出的softmax结果
    :param prob_threshold: 某类的概率大于该threshold则为该类，否则为0
    :return: 按threshold提取标签的结果
    """
    assert len(output.shape) == 2
    assert prob_threshold < 1

    if output.sum().item() > 1:
        output: np.array = F.softmax(output, dim = 1).cpu().numpy()

    output = np.apply_along_axis(partial(judger,prob_threshold = prob_threshold),axis=1, arr=output)

    return output