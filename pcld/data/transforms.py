# -*- coding: utf-8 -*-
import torch
import warnings
from omegaconf.listconfig import ListConfig

from pcld.utils import instantiate_from_config


class SurfaceAxisScale(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True, jitter_scale=0.005):
        assert isinstance(interval, (tuple, list, ListConfig))
        self.interval = interval
        self.jitter = jitter
        self.jitter_scale = jitter_scale

    def __call__(self, surface):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        # print(scaling)
        surface = surface * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale

        if self.jitter:
            surface += self.jitter_scale * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface


class Compose(object):
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def build_transforms(cfg):

    if cfg is None:
        return None

    transforms = []

    for transform_name, cfg_instance in cfg.items():
        transform_instance = instantiate_from_config(cfg_instance)
        transforms.append(transform_instance)
        print(f"Build transform: {transform_instance}")

    transforms = Compose(transforms)

    return transforms

