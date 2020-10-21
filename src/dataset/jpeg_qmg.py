import os
import torch.utils.data as data
from io import BytesIO
from PIL import Image
import numpy as np
import glob
import random
import skimage
import utils.util_image as util
from utils.util_quan import getQM

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
            QMs = getQM(buffer.getvalue())
            imgIn = Image.open(buffer).convert()

        imgGT = np.array(imgGT, copy=False)
        imgIn = np.array(imgIn, copy=False)

        # ========== Quantization map ==========
        if self.colorMode == 'L':
            QM = np.zeros([8,8,1], dtype=np.uint8)
            QM[:,:,0] = QMs[0]
        else:
            QM = np.zeros([8,8,2], dtype=np.uint8)
            QM[:,:,0] = QMs[0]
            QM[:,:,1] = QMs[1]
        assert self.size % 8 == 0, f'The arg \"size\" should be times of 8, but {self.size}.'
        cnt = self.size // 8
        QMimg = np.tile(QM, (cnt,cnt,1))

        if imgIn.ndim == 2:
            imgIn = imgIn[:, :, np.newaxis]
        if imgGT.ndim == 2:
            imgGT = imgGT[:, :, np.newaxis]

        imgGlobal = skimage.transform.resize(imgIn, (112,112), order=3)
        imgIn = np.concatenate((imgIn, QMimg), axis=-1)

        imgIn = util.uint2tensor(imgIn)
        imgGT = util.uint2tensor(imgGT)
        imgGlobal = util.uint2tensor(imgGlobal)

        return [imgIn, imgGlobal], imgGT


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

    def padding8(self, imgGT):
        H, W = imgGT.shape[0:2]
        pad_h = 8 - H % 8 if H % 8 != 0 else 0
        pad_w = 8 - W % 8 if W % 8 != 0 else 0
        if self.colorMode == 'L':
            imgGT = np.pad(imgGT, ((0, pad_h),(0,pad_w)), 'edge')
        else:
            imgGT = np.pad(imgGT, ((0, pad_h),(0,pad_w),(0,0)), 'edge')
        return imgGT

    def __getitem__(self, idx):
        # load GT image
        pathGT = os.path.join(self.imgGTRoot, self.imgNames[idx])
        imgGT = util.imread_uint8(pathGT, mode=self.colorMode)

        # pad the image to multiple of 8
        ori_h, ori_w = imgGT.shape[0:2]
        imgGT = self.padding8(imgGT)

        # ========== Low quality image ==========
        imgGT = Image.fromarray(imgGT)
        with BytesIO() as buffer:
            imgGT.save(buffer, format='JPEG', quality=self.quality)
            QMs = getQM(buffer.getvalue())
            imgIn = Image.open(buffer).convert()

        imgGT = np.array(imgGT, copy=False)
        imgIn = np.array(imgIn, copy=False)
        LQimg = imgIn.copy()

        # ========== Quantization map ==========
        if self.colorMode == 'L':
            QM = np.zeros([8,8,1], dtype=np.uint8)
            QM[:,:,0] = QMs[0]
        else:
            QM = np.zeros([8,8,2], dtype=np.uint8)
            QM[:,:,0] = QMs[0]
            QM[:,:,1] = QMs[1]
        cnt_h, cnt_w = np.array(imgIn.shape[0:2]) // 8
        QMimg = np.tile(QM, (cnt_h, cnt_w, 1))

        if imgIn.ndim == 2:
            imgIn = imgIn[:, :, np.newaxis]
        if imgGT.ndim == 2:
            imgGT = imgGT[:, :, np.newaxis]

        imgGlobal = skimage.transform.resize(imgIn, (112,112), order=3)
        imgIn = np.concatenate((imgIn, QMimg), axis=-1)

        imgIn = util.uint2tensor(imgIn)
        imgGT = util.uint2tensor(imgGT)
        imgGlobal = util.uint2tensor(imgGlobal)

        return [imgIn, imgGlobal], imgGT, LQimg, (ori_h, ori_w)
