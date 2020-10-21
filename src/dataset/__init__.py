from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from torch.utils.data import DataLoader
from importlib import import_module

def buildDataLoader(opt):
    module = import_module('dataset.' + opt.dataloader)

    train_set = module.TrainData(opt)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
        batch_size=opt.batchSize, shuffle=True)

    if opt.validation:
        valid_set = module.ValidData(imgValidRoot=opt.imgValidRoot, 
            colorMode=opt.colorMode, quality=10)
        valid_data_loader = DataLoader(dataset=valid_set, 
            batch_size=max(1, int(pow(opt.size,2)*opt.batchSize/pow(256,2))), shuffle=False)
    else:
        valid_data_loader = None
    
    return training_data_loader, valid_data_loader