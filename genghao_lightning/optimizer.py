from .imports import * 

__all__ = [
    'create_optimizer',
]


def create_optimizer(name: str, 
                     param: dict[str, Any],
                     model: nn.Module) -> optim.Optimizer:
    if name == 'Adam':
        lr = param['lr']
        weight_decay = param.get('weight_decay', 0.)
        
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    else:
        raise AssertionError 
