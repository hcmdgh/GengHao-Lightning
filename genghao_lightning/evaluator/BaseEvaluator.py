__all__ = [
    'BaseEvaluator', 
]


class BaseEvaluator:
    def __init__(self,
                 use_wandb: bool = True):
        self.use_wandb = use_wandb
    
    def eval_train_epoch(self, **kwargs):
        raise NotImplementedError
    
    def eval_val_epoch(self, **kwargs):
        raise NotImplementedError
    
    def eval_test_epoch(self, **kwargs):
        raise NotImplementedError
    
    def eval_train_step(self, **kwargs):
        raise NotImplementedError
    
    def eval_val_step(self, **kwargs):
        raise NotImplementedError
    
    def eval_test_step(self, **kwargs):
        raise NotImplementedError
