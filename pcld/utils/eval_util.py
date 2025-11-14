# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import Union


def compute_mean_iou(pred: Union[torch.Tensor, torch.LongTensor, np.ndarray],
                     labels: Union[torch.LongTensor, np.ndarray],
                     num_classes: int = 3):

    """ Compute mean iou. This function is compatible for both torch.FloatTensor and np.ndarray.

    Args:
        pred (torch.Tensor or torch.LongTensor or np.ndarray): [...]
        labels (torch.LongTensor or np.ndarray): [...]
        num_classes (int): the number of classes.

    Returns:
        miou (float): the mean iou.

    """

    miou = 0
    for class_id in range(num_classes):
        pred_class = pred == class_id
        true_class = labels == class_id

        intersection = (pred_class * true_class).sum() + 0.0
        union = ((pred_class + true_class) > 0).sum() + 1e-5

        iou = intersection / union
        miou += iou

    miou /= num_classes

    return miou


if __name__ == "__main__":
    bs = 10
    num_points = 4096
    n_classes = 3

    pred_ny = np.random.choice(n_classes, (bs, num_points))
    labels_ny = np.random.choice(n_classes, (bs, num_points))

    pred_pt = torch.LongTensor(pred_ny)
    labels_pt = torch.LongTensor(labels_ny)

    miou_np = compute_mean_iou(pred_ny, labels_ny, n_classes)
    miou_pt = compute_mean_iou(pred_pt, labels_pt, n_classes)

    print(miou_np, miou_pt)

