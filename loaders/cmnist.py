from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST as mnist_original
import torchvision.transforms as transforms
from utils.colors import get_colors, get_random_color_variation

import numpy as np


class CMNIST(Dataset):
    '''
    Colored MNIST Dataset
    '''
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            variance: float = 0.02,
            colorize: str = 'fg',
            download: bool = True,
            same_distribution: bool = True,
            class_subset: List[int] = None,
            color_fg_func: Optional[Callable] = get_colors,
    ) -> None:
        super().__init__()
        assert colorize in ['fg', 'bg', 'both'], \
                'colorize must be (fg|bg|both)'

        self.root = root
        self.train = train
        self.variance = variance
        self.colorize = colorize
        self.download = download
        # if not using the same distribution, we remove bias from test
        self.same_distribution = same_distribution

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.mnist = mnist_original(root, train, download=download)
        self.data = self.mnist.data
        self.targets = self.mnist.targets

        self.colors_fg = {}
        self.colors_bg = {}
        colors = color_fg_func()
        for i in range(len(colors)):
            self.colors_fg[i] = colors[i]
            self.colors_bg[i] = colors[(i+1)%len(colors)]

        # if using a subset, get class indices
        self.num_classes = 10
        if class_subset is not None:
            self.num_classes = len(class_subset)
            cls_indices = []
            for cls in class_subset:
                indices = (self.targets == cls).nonzero(as_tuple=False)
                cls_indices.append(indices.squeeze())

            cls_indices = torch.cat(cls_indices, dim=0)
            self.data = self.data[cls_indices]
            self.targets = self.targets[cls_indices]


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[index], int(self.targets[index])

        # mask_fg = img > 0
        # mask_bg = ~mask_fg

        img = img.float() / 255
        img = img.repeat(3, 1, 1)

        if self.train or self.same_distribution:
            # each digit has its own unique color
            col_fg = self.colors_fg[target]
            col_bg = self.colors_bg[target]
        else:
            # we sample from all colors
            color_indices = torch.randperm(len(self.colors_fg))[:2]
            col_fg = self.colors_fg[color_indices[0].item()]
            col_bg = self.colors_bg[color_indices[1].item()]

        col_fg = get_random_color_variation(col_fg, variance=self.variance)
        col_bg = get_random_color_variation(col_bg, variance=self.variance)

        if self.colorize == 'fg':
            img[0, ...] *= col_fg[0]
            img[1, ...] *= col_fg[1]
            img[2, ...] *= col_fg[2]
        elif self.colorize == 'bg':
            img = 1 - img
            img[0, ...] *= col_bg[0]
            img[1, ...] *= col_bg[1]
            img[2, ...] *= col_bg[2]
        else:
            # TODO: find a more elegant solution
            img[0, ...] = img[0, ...]*col_fg[0] + (1-img[0, ...])*col_bg[0]
            img[1, ...] = img[1, ...]*col_fg[1] + (1-img[1, ...])*col_bg[1]
            img[2, ...] = img[2, ...]*col_fg[2] + (1-img[2, ...])*col_bg[2]

        img = transforms.ToPILImage()(img.to(torch.uint8))
        img = self.transform(img)

        return img, target