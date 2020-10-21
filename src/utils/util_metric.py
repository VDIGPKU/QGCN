import math
import numpy as np
from io import BytesIO
from PIL import Image
from .ssim import structural_similarity

SSIM = structural_similarity

def PSNR(x, gt, maxval=255, shave_border=0):
    x = x.astype(np.float)
    gt = gt.astype(np.float)
    height, width = x.shape[:2]
    x = x[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    diff = x - gt
    RMSE = math.sqrt(np.mean(diff ** 2))
    if RMSE == 0:
        return 100
    return 20 * math.log10(maxval / RMSE)


def PSNR_B(x, gt, blockSize=8, maxval=255):
    x = x.astype(np.float)
    gt = gt.astype(np.float)
    height, width = x.shape[:2]

    B = blockSize # block size
    Nh = width
    Nv = height
    
    Nhb = Nv * (Nh/B - 1)
    Nvb = Nh * (Nv/B - 1)
    Nhbc = Nv * (Nh - 1) - Nhb
    Nvbc = Nh * (Nv - 1) - Nvb

    Idx_H = range(1, Nh)
    Idx_Hb = range(B, Nh, B)
    Idx_Hbc = np.setxor1d(Idx_H, Idx_Hb)
    Idx_V = range(1, Nv)
    Idx_Vb = range(B, Nv, B)
    Idx_Vbc = np.setxor1d(Idx_V, Idx_Vb)

    Db = 0
    Dbc = 0
    for i in Idx_Hb:
        Db += np.sum(np.power(x[:,i-1,...] - x[:,i,...], 2))
    for i in Idx_Vb:
        Db += np.sum(np.power(x[i-1,:,...] - x[i,:,...], 2))
    for i in Idx_Hbc:
        Dbc += np.sum(np.power(x[:,i-1,...] - x[:,i,...], 2))
    for i in Idx_Vbc:
        Dbc += np.sum(np.power(x[i-1,:,...] - x[i,:,...], 2))
    Db = Db / (Nhb + Nvb)
    Dbc = Dbc / (Nhbc + Nvbc)

    eta = np.log2(B) / np.log2(np.minimum(Nh, Nv)) if Db > Dbc else 0
    BEF = eta * (Db - Dbc)

    MSE = np.mean((x - gt) ** 2)
    MSE_B = MSE + BEF
    PSNR_B = 10 * np.log10(maxval**2 / MSE_B)
    return PSNR_B
