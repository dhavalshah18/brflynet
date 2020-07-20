import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def compute_loss(out_top, out_bottom, gt_top, gt_bottom):
    loss_mse_top = torch.sum(torch.pow((gt_top - out_top), 2))
    loss_mse_bottom = torch.sum(torch.pow((gt_bottom - out_bottom), 2))

    prod_top = -F.log_softmax(out_top, dim=1) * F.softmax(gt_top, dim=1)
    prod_cor = -F.log_softmax(out_bottom, dim=1) * F.softmax(gt_bottom, dim=1)

    loss_ce_top = torch.sum(torch.sum(torch.sum(torch.sum(prod_top, dim=2), dim=2), dim=0))
    loss_ce_bottom = torch.sum(torch.sum(torch.sum(torch.sum(prod_cor, dim=2), dim=2), dim=0))

    return loss_mse_top + loss_mse_bottom + loss_ce_bottom + loss_ce_top

def compute_loss_ce(out_top, out_bottom, gt_top, gt_bottom):
    if len(gt_top.size()) == 4 and len(gt_bottom.size()) == 4:
        gt_top = gt_top.squeeze(1)
        gt_bottom = gt_bottom.squeeze(1)
        
    assert len(out_top.size()) == 4 and len(out_bottom.size()) == 4
    assert out_top.size(1) == 2 and out_bottom.size(1) == 2
        
    loss_ce_top = nn.CrossEntropyLoss(out_top, gt_top)
    loss_ce_bottom = nn.CrossEntropyLoss(out_bottom, gt_bottom)
    
    return loss_ce_top + loss_ce_bottom
