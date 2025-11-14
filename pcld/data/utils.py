# -*- coding: utf-8 -*-

import torch
import numpy as np


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    # dataset = worker_info.dataset
    # split_size = dataset.num_records // worker_info.num_workers
    # # reset num_records to the true number to retain reliable length information
    # dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
    # current_id = np.random.choice(len(np.random.get_state()[1]), 1)
    # return np.random.seed(np.random.get_state()[1][current_id] + worker_id)

    return np.random.seed(np.random.get_state()[1][0] + worker_id)
