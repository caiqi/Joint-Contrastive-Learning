# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultiCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, k_crops):
        self.base_transform = base_transform
        self.k_crops = k_crops
        assert self.k_crops >= 2

    def __call__(self, x):
        total_out = []
        q = self.base_transform(x)
        total_out.append(q)
        for j in range(self.k_crops - 1):
            k = self.base_transform(x)
            total_out.append(k)
        return total_out
