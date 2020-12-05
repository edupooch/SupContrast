from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.celeba import CelebA as celeba_original
import torchvision.transforms as transforms
import torch

class CelebA(Dataset):

    attr = [
        'Arched_Eyebrows',
        'Big_Lips',
        'Big_Nose',
        'Black_Hair',
        'Blond_Hair',
        'Brown_Hair',
        'Bushy_Eyebrows',
        'Eyeglasses',
        'Gray_Hair',
        'High_Cheekbones',
        'Narrow_Eyes',
        'Oval_Face',
        'Pale_Skin',
        'Pointy_Nose',
        'Rosy_Cheeks',
        'Straight_Hair',
        'Wavy_Hair'
    ]

    def __init__(
        self,
        root: str,
        split: str,
        class_subset: List[str] = None,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        super().__init__()

        if isinstance(class_subset, str) and class_subset == 'all':
            class_subset = self.attr

        if not isinstance(class_subset, list):
            class_subset = list(class_subset)

        if not set(class_subset).issubset(self.attr):
            raise Exception(f'Attribute should be in {self.attr}.')

        if split not in ['train', 'valid', 'test']:
            raise Exception("Split should be in ['train', 'valid', 'test'].")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.dataset = celeba_original(
            root=root,
            split=split,
            target_type='attr',
            download=True
        )

        self.class_index = [idx for idx, val in enumerate(self.dataset.attr_names) if val in class_subset]
        self.bias_index = [idx for idx, val in enumerate(self.dataset.attr_names) if val == 'Male']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx)-> Tuple[torch.Tensor, int]:
        information = self.dataset[idx]
        image = information[0]
        labels = information[1]

        # Image
        image = self.transform(image)

        # Labels
        label = int(labels[self.class_index][0])

        # Bias
        bias = labels[self.bias_index][0]

        return image, label