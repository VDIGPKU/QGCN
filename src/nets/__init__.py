from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from importlib import import_module

def buildModel(opt):
    [netFile, netClass] = opt.net.split('-') if '-' in opt.net else [opt.net, 'Net']
    module = import_module('nets.' + netFile)

    if netFile in ['edsr']:
        model = getattr(module, netClass)(
            n_colors    =   opt.n_colors, 
            n_feats     =   opt.n_feats, 
            n_resblocks =   opt.n_resblocks, 
            res_scale   =   opt.res_scale
        )
    elif netFile in ['qmar']:
        model = getattr(module, netClass)(
            in_channel  =   opt.in_channel, 
            out_channel =   opt.out_channel, 
            n_colors    =   opt.n_colors, 
            n_feats     =   opt.n_feats, 
            n_resblocks =   opt.n_resblocks, 
            res_scale   =   opt.res_scale
        )
    elif netFile in ['qmarg']:
        model = getattr(module, netClass)(
            in_channel  =   opt.in_channel, 
            in_glo_channel = opt.in_glo_channel,
            out_channel =   opt.out_channel, 
            n_colors    =   opt.n_colors, 
            n_feats     =   opt.n_feats, 
            n_resblocks =   opt.n_resblocks, 
            res_scale   =   opt.res_scale
        )
    elif netFile == 'arcnn':
        netClass = 'ARCNN' if netClass == 'Net' else netClass
        model = getattr(module, netClass)(
            n_colors =   opt.n_colors
        )
    else:
        raise RuntimeError('Invalid network name: {}.'.format(netFile))

    return model
