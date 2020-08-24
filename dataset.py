import math
import numbers
import random
from torchvision import datasets, transforms
from PIL import Image, ImageOps


class RandomRotate(object):
    """Rotate the given PIL.Image counter clockwise around its centre by a random degree
    (drawn uniformly) within angle_range. angle_range is a tuple (angle_min, angle_max).
    Empty region will be padded with color specified in fill."""
    def __init__(self, angle_range=(-180,180), fill='black'):
        assert isinstance(angle_range, tuple) and len(angle_range) == 2 and angle_range[0] <= angle_range[1]
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.angle_range = angle_range
        self.fill = fill

    def __call__(self, img):
        angle_min, angle_max = self.angle_range
        angle = angle_min + random.random() * (angle_max - angle_min)
        theta = math.radians(angle)
        w, h = img.size
        diameter = math.sqrt(w * w + h * h)
        theta_0 = math.atan(float(h) / w)
        w_new = diameter * max(abs(math.cos(theta-theta_0)), abs(math.cos(theta+theta_0)))
        h_new = diameter * max(abs(math.sin(theta-theta_0)), abs(math.sin(theta+theta_0)))
        pad = math.ceil(max(w_new - w, h_new - h) / 2)
        img = ImageOps.expand(img, border=int(pad), fill=self.fill)
        img = img.rotate(angle, resample=Image.BICUBIC)
        return img.crop((pad, pad, w + pad, h + pad))
