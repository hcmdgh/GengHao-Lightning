from .BaseEvaluator import * 
from ..imports import * 
from ..metric import * 

from basic_util import * 

__all__ = [
    'UnsupervisedMultiClassClassificationEvaluator', 
]


class UnsupervisedMultiClassClassificationEvaluator(BaseEvaluator):
    def __init__(self,
                 use_wandb: bool = True):
        super().__init__(use_wandb=use_wandb)
        
        self.epoch_to_val_f1_micro: dict[int, float] = dict() 
        self.epoch_to_val_f1_macro: dict[int, float] = dict() 
        self.epoch_to_test_f1_micro: dict[int, float] = dict()
        self.epoch_to_test_f1_macro: dict[int, float] = dict()
        
    def eval_train_epoch(self,
                         *,
                         epoch: int, 
                         loss: FloatScalarTensor, 
                         **useless) -> FloatScalarTensor:
        log_info(f"[Train] Epoch: {epoch}, Loss: {float(loss):.5f}")
        
        if self.use_wandb:
            wandb.log(
                {
                    'Loss': float(loss), 
                }, 
                step = epoch, 
            )

        return loss 
        
    def eval_val_epoch(self,
                       *, 
                       epoch: int, 
                       val_f1_micro: float,
                       val_f1_macro: float,
                       **useless):
        assert epoch not in self.epoch_to_val_f1_micro
        self.epoch_to_val_f1_micro[epoch] = val_f1_micro
        self.epoch_to_val_f1_macro[epoch] = val_f1_macro
        
        best_val_f1_micro_epoch, best_val_f1_micro = max(self.epoch_to_val_f1_micro.items(), key=lambda x: (x[1], -x[0]))
        best_val_f1_macro_epoch, best_val_f1_macro = max(self.epoch_to_val_f1_macro.items(), key=lambda x: (x[1], -x[0]))

        log_info(f"[Val] Epoch: {epoch}, Val F1-Micro: {val_f1_micro:.4f}, Best Val F1-Micro: {best_val_f1_micro:.4f} (in Epoch {best_val_f1_micro_epoch}), Val F1-Macro: {val_f1_macro:.4f}, Best Val F1-Macro: {best_val_f1_macro:.4f} (in Epoch {best_val_f1_macro_epoch})")

        if self.use_wandb:
            wandb.log(
                {
                    'Val F1-Micro': val_f1_micro, 
                    'Val F1-Macro': val_f1_macro, 
                }, 
                step = epoch, 
            )
    
    def eval_test_epoch(self,
                        *, 
                        epoch: int, 
                        test_f1_micro: float,
                        test_f1_macro: float,
                        **useless):
        assert epoch not in self.epoch_to_test_f1_micro
        self.epoch_to_test_f1_micro[epoch] = test_f1_micro
        self.epoch_to_test_f1_macro[epoch] = test_f1_macro
        
        best_test_f1_micro_epoch, best_test_f1_micro = max(self.epoch_to_test_f1_micro.items(), key=lambda x: (x[1], -x[0]))
        best_test_f1_macro_epoch, best_test_f1_macro = max(self.epoch_to_test_f1_macro.items(), key=lambda x: (x[1], -x[0]))

        log_info(f"[Test] Epoch: {epoch}, Test F1-Micro: {test_f1_micro:.4f}, Best Test F1-Micro: {best_test_f1_micro:.4f} (in Epoch {best_test_f1_micro_epoch}), Test F1-Macro: {test_f1_macro:.4f}, Best Test F1-Macro: {best_test_f1_macro:.4f} (in Epoch {best_test_f1_macro_epoch})")

        if self.use_wandb:
            wandb.log(
                {
                    'Test F1-Micro': test_f1_micro, 
                    'Test F1-Macro': test_f1_macro, 
                }, 
                step = epoch, 
            )
    
    def summary(self):
        best_val_f1_micro_epoch, best_val_f1_micro = max(self.epoch_to_val_f1_micro.items(), key=lambda x: (x[1], -x[0]))
        best_val_f1_macro_epoch, best_val_f1_macro = max(self.epoch_to_val_f1_macro.items(), key=lambda x: (x[1], -x[0]))
        best_test_f1_micro_epoch, best_test_f1_micro = max(self.epoch_to_test_f1_micro.items(), key=lambda x: (x[1], -x[0]))
        best_test_f1_macro_epoch, best_test_f1_macro = max(self.epoch_to_test_f1_macro.items(), key=lambda x: (x[1], -x[0]))
        
        wandb.summary['best_val_f1_micro_epoch'] = best_val_f1_micro_epoch
        wandb.summary['best_val_f1_micro'] = best_val_f1_micro
        wandb.summary['best_val_f1_macro_epoch'] = best_val_f1_macro_epoch
        wandb.summary['best_val_f1_macro'] = best_val_f1_macro
        wandb.summary['best_test_f1_micro_epoch'] = best_test_f1_micro_epoch
        wandb.summary['best_test_f1_micro'] = best_test_f1_micro
        wandb.summary['best_test_f1_macro_epoch'] = best_test_f1_macro_epoch
        wandb.summary['best_test_f1_macro'] = best_test_f1_macro
