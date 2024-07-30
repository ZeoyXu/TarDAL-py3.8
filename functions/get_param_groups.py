# from typing import List

# from torch import nn


# def get_param_groups(module) -> tuple[List, List, List]:
#     group = [], [], []
#     bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers
#     for v in module.modules():
#         if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
#             "bias"
#             group[2].append(v.bias)
#         if isinstance(v, bn):
#             "weight (no decay)"
#             group[1].append(v.weight)
#         elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
#             "weight (with decay)"
#             group[0].append(v.weight)
#     return group


from typing import List, Tuple
from torch import nn

def get_param_groups(module) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    bn = [v for k, v in nn.__dict__.items() if isinstance(v, type) and 'Norm' in k]  # normalization layers
    group = [], [], []
    for v in module.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            group[2].append(v.bias)
        if isinstance(v, tuple(bn)):  # 使用tuple(bn)来确保bn是一个元组
            group[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            group[0].append(v.weight)
    return tuple(group)  # 最后返回一个元组