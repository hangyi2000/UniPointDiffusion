# -*- coding: utf-8 -*-

import torch

from typing import List


def add_batch_index(tensor, ids):
    """

    Args:
        tensor (torch.Tensor): (n, c)
        ids (int):

    Returns:
        tensor (torch.Tensor): (n, c + 1)
    """

    batch = torch.full((tensor.shape[0], 1),
                       ids,
                       device=tensor.device,
                       dtype=tensor.dtype)

    tensor = torch.cat([tensor, batch], dim=-1)

    return tensor


def collate_batch_index(tensor_list: List[torch.Tensor]):

    batch_size = len(tensor_list)

    collated_tensor = []

    for i in range(batch_size):
        tensor = add_batch_index(tensor_list[i], i)
        collated_tensor.append(tensor)

    collated_tensor = torch.cat(collated_tensor, dim=0)

    return collated_tensor
