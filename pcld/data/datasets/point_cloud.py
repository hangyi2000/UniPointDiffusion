# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from pcld.data.transforms import build_transforms

import random
from torchvision import transforms
from PIL import Image

import re

category_list_13 = {
    "02691156": 0,
    "02828884": 1,
    "02933112": 2,
    "02958343": 3,
    "03001627": 4,
    "03211117": 5,
    "03636649": 6,
    "03691459": 7,
    "04090263": 8,
    "04256520": 9,
    "04379243": 10,
    "04401088": 11,
    "04530566": 12,
}

category2text = {
    "02691156": "airplane",
    "02747177": "trash bin",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
    "04554684": "washer",
}

class PointCloudDataset_multi_v405(data.Dataset):

    def __init__(self, dataset_name, dataset_folder, split_folder, 
                 split, transform=None, surface_sampling=True,
                 pc_size=2048, surfaces_folder="surfaces", 
                 img_folder='ShapeNetRendering', text_folder=None):

        self.pc_size = pc_size

        self.transform = build_transforms(transform)

        self.dataset_name = dataset_name
        self.split_folder = split_folder
        self.dataset_folder = dataset_folder
        self.surface_folder = os.path.join(self.dataset_folder, surfaces_folder)
        self.img_folder = os.path.join(self.dataset_folder, img_folder)
        if text_folder is not None:
            self.text_folder = os.path.join(self.dataset_folder, text_folder)
        else:
            self.text_folder = None
        self.surface_sampling = surface_sampling

        self.split = split

        self.models = self.read_models_info()

        self.img_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])


    def read_models_info(self):

        model_info_list = []

        split_path = os.path.join(self.split_folder, f"{self.split}.json")
        with open(split_path, "r") as reader:
            meta_info = json.load(reader)
            for category, models in tqdm(meta_info.items()):
                for model_name in models:
                    model_info = {
                        "category": category,
                        "model": model_name,
                    }
                    model_info_list.append(model_info)

        return model_info_list
    
    def create_new_json(self):

        split_path = os.path.join(self.split_folder, f"{self.split}.json")
        with open(split_path, "r") as reader:
            meta_info = json.load(reader)
            new_meta_info = {}
            for key in meta_info:

                model_list = meta_info[key]
                new_model_list = []

                for model_name in model_list:
                    path = os.path.join('./data/ShapeNetRendering', key, model_name)
                    if os.path.exists(path):
                        new_model_list.append(model_name)

                if len(new_model_list):
                    new_meta_info[key] = new_model_list

        for key in new_meta_info:
            print(key+':'+str(len(new_meta_info[key])))

        for key in new_meta_info:
            print(key+':'+str(len(meta_info[key])))

        new_split_folder = './data/shapenet_rendering'
        new_split_path = os.path.join(new_split_folder, f"{self.split}.json")
        with open(new_split_path, 'w') as json_file:
            json.dump(new_meta_info, json_file)


    def __getitem__(self, item):

        rng = np.random.default_rng()

        instance_info = self.models[item]

        category = instance_info["category"]
        model = instance_info["model"]

        pc_path = os.path.join(self.surface_folder, category, '4_pointcloud', model + ".npz")
        with np.load(pc_path) as pc_data:
            surface = pc_data["points"]

            if self.surface_sampling:
                ind_8192 = rng.choice(surface.shape[0], 8192, replace=False)
                surface_8192 = torch.FloatTensor(surface[ind_8192])
                ind = rng.choice(surface.shape[0], self.pc_size, replace=False)
                surface = torch.FloatTensor(surface[ind])

        if self.transform:
            surface, points = self.transform(surface)

        img_path = os.path.join(self.img_folder, category, model, 'rendering', '{:02d}'.format(random.randrange(0, 24)) + '.png')
        img = self.img_transform(Image.open(img_path))
        img = img[:3, :, :]

        if self.text_folder is not None:
            pass
        else:
            text = category2text[category]

        samples = {
            "category": category,
            "class_num": category_list_13[category],
            "model_name": model,
            "surface_pc": surface,
            "surface_pc_8192": surface_8192,
            "image": img,
            "text": text,
            "img_path": img_path
        }

        return samples

    def __len__(self):
        return len(self.models)
    