import os
import argparse
import torch
import numpy as np
import time
import skimage
from PIL import Image
import glob
from io import BytesIO
import utils.util_image as util
from utils.util_metric import PSNR, PSNR_B, SSIM

parser = argparse.ArgumentParser(description="PyTorch Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_1.pth", type=str, help="model path")
parser.add_argument("--realdatapath", default="testdata", type=str, help="test data path")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--GT", action="store_true", help="evaluate GT image dataset")
parser.add_argument("--gtdata", default="LIVE1", type=str, help="GT data name")
parser.add_argument("--quality", default=10, type=int, help="JPEG quality when evaluating GT images")
parser.add_argument("--save", default="", type=str, help="save results when evaluating GT images")
parser.add_argument("--colorMode", default="RGB", type=str, help="L, RGB")


def cutOddSize(img):
    # cut off edge when odd size
    [h,w] = img.shape[0:2]
    if h % 2 == 1:
        img = img[:-1, :, ...]
    if w % 2 == 1:
        img = img[:, :-1, ...]
    return img

def backOddSize(img, oriimg):
    # give back edge when odd size
    [h,w] = img.shape[0:2]
    [h2,w2] = oriimg.shape[0:2]
    if h2-h==1 or w2-w==1:
        imgRes = oriimg.copy()
        imgRes[:h, :w, ...] = img
    else:
        imgRes = img
    return imgRes


# ===== parameters =====
opt = parser.parse_args()
print(opt)
cuda = opt.cuda

if cuda:
    print(f"=> use gpu id: '{opt.gpus}'")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

# ===== loading data =====
dataPath = {
    'DIV2K': '/data/lijw/data/DIV2K/DIV2K_valid_HR/*.png',
    'LIVE1': '/data/lijw/data/LIVE1/GT/*.bmp',
    'BSDS500': '/data/lijw/data/BSDS500/test/*.png'
}
if opt.GT:
    imgGTRoot = dataPath[opt.gtdata]
    fullPaths = sorted(glob.glob(imgGTRoot))
    psnrListR = []
    psnrListL = []
    psnrbListR = []
    psnrbListL = []
    ssimListR = []
    ssimListL = []
    util.mkdir('results')
else:
    fullPaths = sorted(glob.glob(os.path.join(opt.realdatapath, '*.jpg')))
    saveDir = os.path.join(os.path.dirname(fullPaths[0]), 'results')
    util.mkdir(saveDir)

start_time_all = time.time()
gputime = 0

# ===== loading model =====
modelData = torch.load(opt.model)
model = modelData['model']
if cuda:
    model = model.cuda()
else:
    model = model.cpu()
model.eval()

trainOpt = modelData['opt'] if 'opt' in modelData else None
if trainOpt:
    if opt.colorMode != trainOpt['colorMode']:
        print(f'WARNING: The color mode of this model is {trainOpt["colorMode"]}, not {opt.colorMode}.')
        opt.colorMode = trainOpt['colorMode']

for i in range(len(fullPaths)):
    currStartTime = time.time()
    print(f'\n[{i+1}/{len(fullPaths)}]', os.path.basename(fullPaths[i]))
    if opt.GT:
        GTImg = util.imread_uint8(fullPaths[i], mode=opt.colorMode)
        GTImg = Image.fromarray(GTImg)
        with BytesIO() as buffer:
            GTImg.save(buffer, format='JPEG', quality=opt.quality)
            currImg = Image.open(buffer).convert()
        GTImg = np.array(GTImg)
        currImg = np.array(currImg)
    else:
        currImg = util.imread_uint8(fullPaths[i], mode=opt.colorMode)

    LQImg = currImg.copy()

    start_time = time.time()
    currImg = cutOddSize(currImg)
    currImg = util.uint2tensor(currImg)
    currImg = util.tensor3to4(currImg)
    if cuda:
        currImg = currImg.cuda()
    HQImg = model(currImg)
    HQImg = util.tensor2uint(HQImg)
    HQImg = backOddSize(HQImg, LQImg)

    elapsed_time = time.time() - start_time
    gputime += elapsed_time

    if opt.GT:
        psnrR = PSNR(HQImg, GTImg)
        psnrL = PSNR(LQImg, GTImg)
        ssimR = SSIM(HQImg, GTImg, data_range=255, win_size=8)
        ssimL = SSIM(LQImg, GTImg, data_range=255, win_size=8)
        psnrbR = PSNR_B(HQImg, GTImg, 8)
        psnrbL = PSNR_B(LQImg, GTImg, 8)
        print(f'PSNR: LQ: {psnrL:.4f}, HQ: {psnrR:.4f}, Increment: {(psnrR - psnrL):.4f}')
        print(f'SSIM: LQ: {ssimL:.4f}, HQ: {ssimR:.4f}, Increment: {(ssimR - ssimL):.4f}')
        print(f'PSNR-B: LQ: {psnrbL:.4f}, HQ: {psnrbR:.4f}, Increment: {(psnrbR - psnrbL):.4f}')
        psnrListR.append(psnrR)
        psnrListL.append(psnrL)
        ssimListR.append(ssimR)
        ssimListL.append(ssimL)
        psnrbListR.append(psnrbR)
        psnrbListL.append(psnrbL)
        if opt.save:
            util.mkdir(os.path.join('./results/', opt.save))
            skimage.io.imsave(os.path.join('./results/', opt.save, os.path.basename(fullPaths[i])[:-4] + '-HQ.png'), HQImg)
            skimage.io.imsave(os.path.join('./results/', opt.save, os.path.basename(fullPaths[i])[:-4] + '-LQ.png'), LQImg)
    else:
        skimage.io.imsave(os.path.join(saveDir, os.path.basename(fullPaths[i])[:-4] + '.png'), HQImg)
    currSpendTime = time.time() - currStartTime
    print(f'Time cost: {currSpendTime:.4f}s')

print('\n【Average of all images:】')
if opt.GT:
    psnrMeanR = np.mean(psnrListR)
    psnrMeanL = np.mean(psnrListL)
    ssimMeanR = np.mean(ssimListR)
    ssimMeanL = np.mean(ssimListL)
    psnrbMeanR = np.mean(psnrbListR)
    psnrbMeanL = np.mean(psnrbListL)
    print(f'PSNR: LQ: {psnrMeanL:.4f}, HQ: {psnrMeanR:.4f}, Increment: {(psnrMeanR - psnrMeanL):.4f}')
    print(f'SSIM: LQ: {ssimMeanL:.4f}, HQ: {ssimMeanR:.4f}, Increment: {(ssimMeanR - ssimMeanL):.4f}')
    print(f'PSNR-B: LQ: {psnrbMeanL:.4f}, HQ: {psnrbMeanR:.4f}, Increment: {(psnrbMeanR - psnrbMeanL):.4f}')
elapsed_time_all = time.time() - start_time_all
print(f'All time: {elapsed_time_all:.4f}s, primary time: {gputime:.4f}s')
print(f'============= END ==============')
