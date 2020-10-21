import torch.nn as nn
from importlib import import_module

class Loss(nn.modules.loss._Loss):
    def __init__(self, opt):
        super(Loss, self).__init__()
        self.loss = []
        if opt.loss == 'L1':
            self.loss.append({
                'type': 'L1',
                'weight': 1.0,
                'function': nn.L1Loss()}
            )
        elif opt.loss == 'L2':
            self.loss.append({
                'type': 'L2',
                'weight': 1.0,
                'function': nn.MSELoss()}
            )
        else:
            raise RuntimeError('Invalid loss function: {}.'.format(opt.loss))

    def forward(self, x, gt):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](x, gt)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
        loss_sum = sum(losses)
        return loss_sum
