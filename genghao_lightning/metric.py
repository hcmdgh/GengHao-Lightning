from .imports import * 

from basic_util import * 
from sklearn.metrics import f1_score

__all__ = [
    'calc_f1_micro',  
    'calc_f1_macro',  
]


def calc_f1_micro(pred: Union[ndarray, Tensor],
                  target: Union[ndarray, Tensor]) -> float:
    if isinstance(pred, Tensor):
        pred = pred.detach().cpu().numpy() 
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy() 
    
    # 第1种情况-多分类单标签：pred = int[N], target = int[N]
    if pred.ndim == 1:
        assert pred.dtype == target.dtype == np.int64 
        N = len(pred)
        assert pred.shape == target.shape == (N,) 
        
        f1_micro = f1_score(y_pred=pred, y_true=target, average='micro')

        return float(f1_micro)
     
    # 第2种情况-多分类单标签：input = float[N, D], target = int[N]
    elif pred.ndim == 2 and target.ndim == 1:
        assert pred.dtype == np.float32 and target.dtype == np.int64  
        N, D = pred.shape 
        assert target.shape == (N,)
        
        pred = np.argmax(pred, axis=-1) 
        
        f1_micro = f1_score(y_pred=pred, y_true=target, average='micro')
        
        return float(f1_micro)
    
    # 第3种情况-多分类多标签：input = int[N, D], target = int[N, D]
    elif pred.ndim == 2 and target.ndim == 2:
        N, D = pred.shape 
        assert pred.dtype == target.dtype == np.int64 
        assert target.shape == (N, D)
        
        f1_micro = f1_score(y_pred=pred, y_true=target, average='micro')
        
        return float(f1_micro)
     
    else:
        raise AssertionError


def calc_f1_macro(pred: Union[ndarray, Tensor],
                  target: Union[ndarray, Tensor]) -> float:
    if isinstance(pred, Tensor):
        pred = pred.detach().cpu().numpy() 
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy() 
    
    # 第1种情况-多分类单标签：pred = int[N], target = int[N]
    if pred.ndim == 1:
        assert pred.dtype == target.dtype == np.int64 
        N = len(pred)
        assert pred.shape == target.shape == (N,) 
        
        f1_macro = f1_score(y_pred=pred, y_true=target, average='macro')

        return float(f1_macro)
     
    # 第2种情况-多分类单标签：input = float[N, D], target = int[N]
    elif pred.ndim == 2 and target.ndim == 1:
        assert pred.dtype == np.float32 and target.dtype == np.int64  
        N, D = pred.shape 
        assert target.shape == (N,)
        
        pred = np.argmax(pred, axis=-1) 
        
        f1_macro = f1_score(y_pred=pred, y_true=target, average='macro')
        
        return float(f1_macro)
    
    # 第3种情况-多分类多标签：input = int[N, D], target = int[N, D]
    elif pred.ndim == 2 and target.ndim == 2:
        N, D = pred.shape 
        assert pred.dtype == target.dtype == np.int64 
        assert target.shape == (N, D)
        
        f1_macro = f1_score(y_pred=pred, y_true=target, average='macro')
        
        return float(f1_macro)
     
    else:
        raise AssertionError
