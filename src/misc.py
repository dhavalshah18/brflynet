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
    
def dice_coeff_2d(output, target, smooth=1):
    probs = F.softmax(output, dim=1)
    _, pred = torch.max(probs, 1)

    if len(target.size()) == 4:
        target = target.squeeze(1)
    
    target = F.one_hot(target.long(), num_classes=2)
    pred = F.one_hot(pred.long(), num_classes=2)

    dim = tuple(range(1, len(pred.size())-1))
    intersection = torch.sum(target * pred, dim=dim, dtype=torch.float)
    union = torch.sum(target, dim=dim, dtype=torch.float) + torch.sum(pred, dim=dim, dtype=torch.float)
    
    dice = torch.mean((2. * intersection + smooth)/(union + smooth), dtype=torch.float)
        
    return dice

def dice_coeff_2arm(out_top, out_bottom, gt_top, gt_bottom):
    dice_top = dice_coeff_2d(out_top, gt_top)
    dice_bottom = dice_coeff_2d(out_bottom, gt_bottom)
    
    return (dice_top + dice_bottom) / 2

def loss_mse_ce(out_top, out_bottom, gt_top, gt_bottom):
    loss_mse_top = nn.MSELoss()(out_top, gt_top)
    loss_mse_bottom = nn.MSELoss()(out_bottom, gt_bottom)
    
    loss_ce_total = loss_ce(out_top, out_bottom, gt_top, gt_bottom)
    
    return loss_mse_top + loss_mse_bottom + loss_ce_total

def loss_ce(out_top, out_bottom, gt_top, gt_bottom):
    if len(gt_top.size()) == 4 and len(gt_bottom.size()) == 4:
        gt_top = gt_top.squeeze(1)
        gt_bottom = gt_bottom.squeeze(1)
        
    assert len(out_top.size()) == 4 and len(out_bottom.size()) == 4
    assert out_top.size(1) == 2 and out_bottom.size(1) == 2
    
    loss_ce_top = nn.CrossEntropyLoss()(out_top, gt_top)
    loss_ce_bottom = nn.CrossEntropyLoss()(out_bottom, gt_bottom)
    
    return loss_ce_top + loss_ce_bottom

def dice_loss_2arm(out_top, out_bottom, gt_top, gt_bottom):    
    dice_total_top = dice_loss_2d(out_top, gt_top)
    dice_total_bottom = dice_loss_2d(out_bottom, gt_bottom)
    
    return dice_total_top + dice_total_bottom

def dice_loss_2d(output, target):
    if len(target.size()) == 4:
        target = target.squeeze(1) 
    
    target = F.one_hot(target, num_classes=2)
    target = target.permute(0, 3, 1, 2)
    
    pred = F.softmax(output, dim=1)
    
    dim = tuple(range(2, len(pred.size())))
    
    num = pred * target         # b,c,h,w
    num = torch.sum(num, dim=dim) # b, c
    
    den1 = target**2
    den1 = torch.sum(den1, dim=dim)

    den2 = pred**2
    den2 = torch.sum(den2, dim=dim) 
    
    dice = 2.*(num + 1.)/(den1 + den2 + 1.)
    dice_eso = dice[:,1:]       # ignore background dice val, and take the foreground

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0) # divide by batch_sz
    
    return dice_total
