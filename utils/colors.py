from typing import List

from colorsys import hsv_to_rgb
import math
import random
import torch


def get_colors() -> List[List[int]]:
    '''
    Extracted from 'Learning Not to Learn: Training Deep Neural Networks with
    Biased Data' (Supplementary Material)

    https://arxiv.org/abs/1812.10352
    '''
    colors = [
            [220,  20,  60], # (0) Crimson
            [  0, 128, 128], # (1) Teal
            [253, 233,  16], # (2) Lemon
            [  0, 149, 182], # (3) Bondi Blue
            [237, 145,  33], # (4) Carrot Orange
            [145,  30, 188], # (5) Strong Violet
            [ 70, 240, 240], # (6) Cyan
            [250, 197, 187], # (7) Your Pink
            [210, 245,  60], # (8) Lime
            [128,   0,   0], # (9) Maroon
    ]
    return colors


def get_colors_binary() -> List[List[int]]:
    colors = [
            [100, 100, 100],
            [200, 200, 200],
    ]
    return colors


def generate_colors(num_colors: int) -> List[List[int]]:
    golden_ratio_conjugate = (1 + math.sqrt(5)) / 2

    colors = []
    hue = random.random()  # use random start value

    for i in range(num_colors):
        hue += golden_ratio_conjugate * (i / (5 * random.random()))
        hue = hue % 1
        color_temp = [round(x * 256) for x in hsv_to_rgb(hue, 0.5, 0.95)]
        colors.append(color_temp)

    return colors


def get_random_color_variation(
        color: List[int],
        variance: float
    ) -> List[int]:
    '''
    Extracted from 'Learning Not to Learn: Training Deep Neural Networks with
    Biased Data' (Supplementary Material)

    https://arxiv.org/abs/1812.10352
    '''
    color = [x/255 for x in color]

    sampled_color = []
    for c in color:
        while True:
            val = torch.normal(mean=torch.Tensor([c]),
                    std=torch.Tensor([math.sqrt(variance)])).item()
            if val > 0 and val < 1:
                sampled_color.append(val)
                break

    sampled_color = [int(x*255) for x in sampled_color]
    return sampled_color