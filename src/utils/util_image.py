import os
import numpy as np
import random
import numbers
import skimage
from skimage import io, color
import torch

# read uint8 image from path
def imread_uint8(imgpath, mode='RGB'):
    '''
    mode: 'RGB', 'gray', 'Y', 'L'.
    'Y' and 'L' mean the Y channel of YCbCr.
    '''
    if mode == 'RGB':
        img = io.imread(imgpath)
    elif mode == 'gray':
        img = io.imread(imgpath, as_gray=True)
        img = skimage.img_as_ubyte(img)
    elif mode in ['Y','L']: 
        # Y channel of YCbCr
        # Note: The skimage.color.rgb2ycbcr() function is the same with that of matlab,
        # PIL.Image.convert('YCbCr') is not.
        img = io.imread(imgpath)
        if img.ndim == 3:
            img = color.rgb2ycbcr(img)[:,:,0]
            img = img.round().astype(np.uint8)
    return img

def augment_img(img, mode='8'):
    '''flip and/or rotate the image randomly'''
    if mode == '2':
        mode = random.randint(0, 1)
    elif mode == '4':
        mode = random.randint(0, 3)
    elif mode == '8':
        mode = random.randint(0, 7)
    else:
        mode = 0

    if mode == 0:
        return img
    elif mode == 1:
        return np.fliplr(img)
    elif mode == 2:
        return np.rot90(img, k=2)
    elif mode == 3:
        return np.fliplr(np.rot90(img, k=2))
    elif mode == 4:
        return np.rot90(img, k=1)
    elif mode == 5:
        return np.fliplr(np.rot90(img, k=1))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.fliplr(np.rot90(img, k=3))
 

def random_crop(img, size):
    '''crop image patch randomly'''
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    h, w = img.shape[0:2]
    ph, pw = size
    rnd_h = random.randint(0, h - ph)
    rnd_w = random.randint(0, w - pw)
    img_patch = img[rnd_h:rnd_h + ph, rnd_w:rnd_w + pw, ...]
    return img_patch


def uint2tensor(img, normalized=True):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    img = skimage.img_as_float32(img)
    if normalized:
        img = (img - 0.5) / 0.5
    img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
    return img


def tensor2uint(img, normalized=True):
    img = img.data.squeeze().cpu().numpy().astype(np.float32)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    elif img.ndim == 4:
        img = img.transpose(0, 2, 3, 1)
    if normalized:
        img = img * 0.5 + 0.5
    img = img.clip(0, 1) * 255
    img = img.round().astype(np.uint8)
    return img


def tensor3to4(tensor):
    return tensor.unsqueeze(0)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
