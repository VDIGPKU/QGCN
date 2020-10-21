import os
import torch.utils.data as data
from io import BytesIO
from PIL import Image
import numpy as np
import glob
import random
import utils.util_image as util

class TrainData(data.Dataset):
    def __init__(self, opt):
        super(TrainData, self).__init__()
        fullPaths = sorted(glob.glob(os.path.join(opt.imgGTRoot, '*')))
        self.imgNames = [os.path.basename(fp) for fp in fullPaths]
        self.imgGTRoot = opt.imgGTRoot
        self.quality = eval(opt.quality) if isinstance(opt.quality, str) else opt.quality
        self.colorMode = opt.colorMode
        self.size = opt.size

    def __len__(self):
        return len(self.imgNames)

    def __getitem__(self, idx):
        # load GT image
        pathGT = os.path.join(self.imgGTRoot, self.imgNames[idx])
        imgGT = util.imread_uint8(pathGT, mode=self.colorMode)

        # ========== High quality patch ==========
        imgGT = util.random_crop(imgGT, self.size)
        imgGT = util.augment_img(imgGT)

        # ========== Low quality patch ==========
        imgGT = Image.fromarray(imgGT)
        assert self.quality[0] <= self.quality[1], f'Illegal quality range {self.quality}'
        quality = random.randint(self.quality[0], self.quality[1])
        with BytesIO() as buffer:
            imgGT.save(buffer, format='JPEG', quality=quality)
            imgIn = Image.open(buffer).convert()

        imgGT = np.array(imgGT, copy=False)
        imgIn = np.array(imgIn, copy=False)

        imgIn = util.uint2tensor(imgIn)
        imgGT = util.uint2tensor(imgGT)

        return imgIn, imgGT


class ValidData(data.Dataset):
    def __init__(self, imgValidRoot, quality=10, colorMode='RGB'):
        super(ValidData, self).__init__()
        fullPaths = sorted(glob.glob(os.path.join(imgValidRoot, '*')))
        self.imgNames = [os.path.basename(fp) for fp in fullPaths]
        self.imgGTRoot = imgValidRoot
        self.quality = quality
        self.colorMode = colorMode

    def __len__(self):
        return len(self.imgNames)

    def __getitem__(self, idx):
        # load GT image
        pathGT = os.path.join(self.imgGTRoot, self.imgNames[idx])
        imgGT = util.imread_uint8(pathGT, mode=self.colorMode)

        # ========== Low quality image ==========
        imgGT = Image.fromarray(imgGT)
        with BytesIO() as buffer:
            imgGT.save(buffer, format='JPEG', quality=self.quality)
            imgIn = Image.open(buffer).convert()
            
        imgGT = np.array(imgGT, copy=False)
        imgIn = np.array(imgIn, copy=False)

        imgIn = util.uint2tensor(imgIn)
        imgGT = util.uint2tensor(imgGT)

        return imgIn, imgGT
