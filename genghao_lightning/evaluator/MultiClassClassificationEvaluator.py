from .BaseEvaluator import * 
from ..imports import * 
from ..metric import * 
from ..device import * 

from basic_util import * 

__all__ = [
    'MultiClassClassificationEvaluator', 
]


class MultiClassClassificationEvaluator(BaseEvaluator):
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
                         pred: Optional[FloatTensor] = None,
                         target: Optional[IntTensor] = None,
                         loss: Optional[FloatScalarTensor] = None, 
                         **useless) -> FloatScalarTensor:
        if loss is not None:
            loss = loss 
        elif pred is not None and target is not None:
            loss = F.cross_entropy(input=pred, target=target)
        else:
            raise AssertionError
            
        log_info(f"[Train] Epoch: {epoch}, Loss: {float(loss):.5f}")
        
        if self.use_wandb:
            wandb.log(
                {
                    'Loss': float(loss), 
                }, 
                step = epoch, 
            )

        return loss 
    
    def eval_train_step(self,
                        *,
                        epoch: int, 
                        step: int, 
                        num_steps: Optional[int] = None, 
                        pred: Optional[FloatTensor] = None,
                        target: Optional[IntTensor] = None,
                        loss: Optional[FloatScalarTensor] = None, 
                        **useless) -> FloatScalarTensor:
        if loss is not None:
            loss = loss 
        elif pred is not None and target is not None:
            loss = F.cross_entropy(input=pred, target=target)
        else:
            raise AssertionError
            
        if num_steps is None:
            log_info(f"[Train] Epoch: {epoch}, Step: {step}, Loss: {float(loss):.5f}")
        else:
            log_info(f"[Train] Epoch: {epoch}, Step: {step}/{num_steps}, Loss: {float(loss):.5f}")
        
        return loss 
        
    def eval_val_epoch(self,
                       *, 
                       epoch: int, 
                       pred: FloatArrayTensor,
                       target: IntArrayTensor,
                       **useless):
        val_f1_micro = calc_f1_micro(pred=pred, target=target)
        val_f1_macro = calc_f1_macro(pred=pred, target=target)
        
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
                        pred: FloatArrayTensor,
                        target: IntArrayTensor,
                        **useless):
        test_f1_micro = calc_f1_micro(pred=pred, target=target)
        test_f1_macro = calc_f1_macro(pred=pred, target=target)
        
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
    
    def eval_val_steps_in_one_epoch(self,
                                    *, 
                                    epoch: int, 
                                    step_data_dict: dict[int, dict[str, Any]], 
                                    **useless):
        step_data_dict = to_numpy(step_data_dict)
        
        pred_list = []
        target_list = []
        
        for data in step_data_dict.values():
            pred_list.append(data['pred'])
            target_list.append(data['target'])

        full_pred = np.concatenate(pred_list, axis=0)
        full_target = np.concatenate(target_list, axis=0)

        return self.eval_val_epoch(
            epoch = epoch, 
            pred = full_pred,
            target = full_target, 
        )
    
    def eval_test_steps_in_one_epoch(self,
                                     *, 
                                     epoch: int, 
                                     step_data_dict: dict[int, dict[str, Any]], 
                                     **useless):
        step_data_dict = to_numpy(step_data_dict)
        
        pred_list = []
        target_list = []
        
        for data in step_data_dict.values():
            pred_list.append(data['pred'])
            target_list.append(data['target'])

        full_pred = np.concatenate(pred_list, axis=0)
        full_target = np.concatenate(target_list, axis=0)

        return self.eval_test_epoch(
            epoch = epoch, 
            pred = full_pred,
            target = full_target, 
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
