import os
import time
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import nets
import dataset
import loss
from params import opt
from utils.util_metric import PSNR
import utils.util_image as util


def main():
    print(opt)
    start_epoch = 0

    cuda = opt.cuda
    if cuda:
        print(f"===> GPU id: '{opt.gpus}'")
        device_ids = eval(opt.gpus) if isinstance(opt.gpus, str) else opt.gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str,device_ids))
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000) if opt.seed == 0 else opt.seed
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    training_data_loader, valid_data_loader = dataset.buildDataLoader(opt)

    print("===> Building model")
    model = nets.buildModel(opt)
    
    print("===> Building Loss")
    criterion = loss.Loss(opt)

    print("===> Setting GPU")
    if cuda:
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(len(device_ids))))
        model = model.cuda()
        criterion = criterion.cuda()

    # resume training from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"=> loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print(f"=> no checkpoint found at '{opt.resume}'")

    # copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print(f"=> loading model '{opt.pretrained}'")
            checkpoint = torch.load(opt.pretrained)
            model.load_state_dict(checkpoint['model'].state_dict())
        else:
            print(f"=> no model found at '{opt.pretrained}'")

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1,
                                          last_epoch=start_epoch-1)

    print("===> Training")
    for epoch in range(start_epoch, opt.nEpochs):
        print("Epoch={}, lr={}".format(epoch+1, optimizer.param_groups[0]["lr"]))
        train(training_data_loader, optimizer, model, criterion, epoch)
        if opt.validation:
            validate(valid_data_loader, model, epoch)
        save_checkpoint(model, epoch)
        scheduler.step()


def trainPrepare(batch):
    if isinstance(batch[0], torch.Tensor):
        inputImg = batch[0].requires_grad_()
        if opt.cuda:
            inputImg = inputImg.cuda()
        source = inputImg
    else:
        source = [x.requires_grad_() for x in batch[0]]
        if opt.cuda:
            source = [x.cuda() for x in source]
    
    target = batch[1]
    if opt.cuda:
        target = target.cuda()
    return source, target


def train(training_data_loader, optimizer, model, criterion, epoch):
    model.train()
    currtime = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):
        source, target = trainPrepare(batch)
        output = model(source)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0 or iteration == len(training_data_loader):
            usetime = time.time() - currtime
            currtime = time.time()
            print(f"===> Epoch[{epoch+1}]({iteration}/{len(training_data_loader)}): Loss: {loss.item():.6f}, Time: {usetime:.4f}")


def validPrepare(batch):
    if isinstance(batch[0], torch.Tensor):
        lrImg = batch[0]
        if opt.cuda:
            lrImg = lrImg.cuda()
        source = lrImg
    else:
        source = batch[0]
        if opt.cuda:
            source = [x.cuda() for x in source]
    
    target = batch[1]
    return source, target


def validate(valid_data_loader, model, epoch):
    model.eval()
    currtime = time.time()
    psnrList = []
    for iteration, batch in enumerate(valid_data_loader, 1):
        source, GTImg = validPrepare(batch)
        HQImg = model(source)
        HQImg = util.tensor2uint(HQImg)
        GTImg = util.tensor2uint(GTImg)
        for i in range(HQImg.shape[0]):
            cHQImg = np.squeeze(HQImg[i])
            cGTImg = np.squeeze(GTImg[i])
            psnr = PSNR(cHQImg, cGTImg)
            psnrList.append(psnr)
    psnrMean = np.mean(psnrList)
    usetime = time.time() - currtime
    print(f"===> Epoch[{epoch+1}] Validation: PSNR: {psnrMean:.4f}, Time: {usetime:.4f}")


def save_checkpoint(model, epoch):
    outdir = os.path.join("checkpoint/", opt.savedir)
    util.mkdir(outdir)
    modelSavePath = os.path.join(outdir, f"model_epoch_{epoch+1}.pth")
    state = {"epoch": epoch, "model": model, "opt": opt.__dict__}
    torch.save(state, modelSavePath)
    print(f"Checkpoint saved to {modelSavePath}")


if __name__ == "__main__":
    main()
