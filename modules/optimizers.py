import math
import torch
from torch import optim


def build_optimizer(args, model):
    if args['optim'] == 'AdamW':
        optimizer = optim.Adam(model.parameters(), lr=args['pt_lr'], weight_decay=args['weight_decay'],
                               amsgrad=args['amsgrad'])
    elif args['optim'] == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=args['pt_lr'], weight_decay=args['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args['pt_lr'], weight_decay=args['weight_decay'])
    return optimizer


def build_two_stage_optimizer(args, model):
    if args['task'] != 'finetune':
        if args['optim'] == 'AdamW':
            optimizer = optim.Adam(model.parameters(), lr=args['pt_lr'], weight_decay=args['weight_decay'],
                                   amsgrad=args['amsgrad'])
        elif args['optim'] == 'RAdam':
            optimizer = optim.RAdam(model.parameters(), lr=args['pt_lr'], weight_decay=args['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=args['pt_lr'], weight_decay=args['weight_decay'])
    else:
        pretrain_main_params, finetune_main_params = [], []
        for name, param in model.named_parameters():
            # if 'text_decoder' in name or 'visual_self_atten_layers' in name or 'multimodal_fusion_layers' in name:
            if 'text_decoder' in name or 'visual_self_atten_layers' in name or 'multimodal_fusion_layers' in name or 'visual_head' in name or "text_head" in name:
                finetune_main_params.append(param)
            else:
                pretrain_main_params.append(param)
        if args['optim'] == 'AdamW':
            optimizer = optim.Adam(
                [{'params': pretrain_main_params, 'lr': args['pt_lr']},
                 {'params': finetune_main_params, 'lr': args['ft_lr']}],
                weight_decay=args['weight_decay'],
                amsgrad=args['amsgrad']
            )
        elif args['optim'] == 'RAdam':
            optimizer = optim.RAdam(
                [{'params': pretrain_main_params, 'lr': args['pt_lr']},
                 {'params': finetune_main_params, 'lr': args['ft_lr']}],
                weight_decay=args['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                [{'params': pretrain_main_params, 'lr': args['pt_lr']},
                 {'params': finetune_main_params, 'lr': args['ft_lr']}],
                weight_decay=args['weight_decay']
            )
    return optimizer


def build_lr_scheduler(args, optimizer):
    # if args['epochs'] <= 4:
    #     T_max = args['epochs']
    # else:
    #     T_max = args['epochs'] // 2
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args['lr'] / 100)
    if args['lr_scheduler'] == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args['monitor_mode'])

    return lr_scheduler
