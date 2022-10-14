from .imports import * 

from basic_util import * 

__all__ = [
    'MulticlassAccuracyMetric', 
    'calc_acc',  
]


def calc_acc(input: Union[ndarray, Tensor],
             target: Union[ndarray, Tensor]) -> float:
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy() 
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy() 
    
    # 第1种情况-多分类单标签：input = int[N], target = int[N]
    # 第2种情况-多分类单标签：input = float[N, D], target = int[N]
    # 第3种情况-多分类多标签：input = int[N, D], target = int[N, D]
    
    # 第1种情况-多分类单标签：input = int[N], target = int[N]
    if input.ndim == 1:
        assert input.dtype == target.dtype == np.int64 
        N = len(input)
        assert input.shape == target.shape == (N,) 
        
        acc = (input == target).mean() 
        
        return float(acc) 
     
    # 第2种情况-多分类单标签：input = float[N, D], target = int[N]
    elif input.ndim == 2 and target.ndim == 1:
        assert input.dtype == np.float32 and target.dtype == np.int64  
        N, D = input.shape 
        assert target.shape == (N,)
        
        input = np.argmax(input, axis=-1) 
        
        acc = (input == target).mean() 
        
        return float(acc) 
    
    # 第3种情况-多分类多标签：input = int[N, D], target = int[N, D]
    elif input.ndim == 2 and target.ndim == 2:
        N, D = input.shape 
        assert input.dtype == target.dtype == np.int64 
        assert target.shape == (N, D)
        
        acc = np.all(input == target, axis=-1).mean() 
        
        return float(acc) 
     
    else:
        raise AssertionError


class MulticlassAccuracyMetric(nn.Module):
    def __init__(self,
                 val_or_test: str):
        super().__init__()
        
        self.val_or_test = val_or_test
        
        self.step_pred_list = []
        self.step_label_list = []

        self.acc_epoch_dict: dict[int, float] = dict() 
        
    def record_step(self,
                    y_pred: FloatArrayTensor,
                    y_true: IntArrayTensor):
        if isinstance(y_pred, Tensor):
            y_pred = y_pred.detach().cpu().numpy() 
        if isinstance(y_true, Tensor):
            y_true = y_true.detach().cpu().numpy() 
            
        self.step_pred_list.append(y_pred)
        self.step_label_list.append(y_true)
            
    def summary_all_step(self,
                         epoch: int):
        full_y_pred = np.concatenate(self.step_pred_list, axis=0)
        full_y_true = np.concatenate(self.step_label_list, axis=0)
        self.step_pred_list.clear() 
        self.step_label_list.clear()
        
        acc = calc_acc(input=full_y_pred, target=full_y_true)
        self.acc_epoch_dict[epoch] = acc 
        
        sort_res = sorted(self.acc_epoch_dict.items(), key=lambda x: (-x[1], x[0]))
        best_acc_epoch, best_acc = sort_res[0]
        
        if self.val_or_test == 'val':
            log_info(f"[Val] Epoch: {epoch}, Val Acc: {acc:.4f} Best Val Acc: {best_acc:.4f} (in Epoch {best_acc_epoch})")
        elif self.val_or_test == 'test':
            log_info(f"[Test] Epoch: {epoch}, Test Acc: {acc:.4f} Best Test Acc: {best_acc:.4f} (in Epoch {best_acc_epoch})")
        else:
            raise AssertionError
        
    def record_epoch(self,
                     epoch: int, 
                     y_pred: FloatArrayTensor,
                     y_true: IntArrayTensor):
        self.step_pred_list.clear() 
        self.step_label_list.clear()
                     
        acc = calc_acc(input=y_pred, target=y_true)
        self.acc_epoch_dict[epoch] = acc 
        
        sort_res = sorted(self.acc_epoch_dict.items(), key=lambda x: (-x[1], x[0]))
        best_acc_epoch, best_acc = sort_res[0]
        
        if self.val_or_test == 'val':
            log_info(f"[Val] Epoch: {epoch}, Val Acc: {acc:.4f} Best Val Acc: {best_acc:.4f} (in Epoch {best_acc_epoch})")
        elif self.val_or_test == 'test':
            log_info(f"[Test] Epoch: {epoch}, Test Acc: {acc:.4f} Best Test Acc: {best_acc:.4f} (in Epoch {best_acc_epoch})")
        else:
            raise AssertionError
