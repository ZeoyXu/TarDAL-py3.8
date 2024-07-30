from typing import Tuple, List, Optional

from torch.optim import Optimizer, AdamW, Adam, SGD

from config import ConfigDict


def smart_optimizer(config: ConfigDict, param_group: Tuple[List, List, List], lr: Optional[float] = None) -> Optimizer:
    if lr is not None:
        config.lr_i = lr
    groups = [
        {'params': param_group[0], 'lr': config.lr_i, 'weight_decay': config.weight_decay},
        {'params': param_group[1], 'lr': config.lr_i, 'weight_decay': 0},
    ]
    if config.name == 'sgd':
        opt = SGD(param_group[2], lr=config.lr_i, momentum=config.momentum, nesterov=True)
    elif config.name == 'adam':
        opt = Adam(param_group[2], lr=config.lr_i, betas=(config.momentum, 0.999))
    elif config.name == 'adamw':
        opt = AdamW(param_group[2], lr=config.lr_i, betas=(config.momentum, 0.999), weight_decay=0)
    else:
        opt = None
        assert NotImplemented, f'unsupported optimizer: {config.name}'
    opt.add_param_group(groups[0])
    opt.add_param_group(groups[1])
    return opt
