from ..imports import * 
from ..device import * 
from ..dataloader import *
from ..optimizer import * 
from ..metric import * 

from basic_util import * 

__all__ = [
    'UnsupervisedRepresentationLearningTask',
]


class UnsupervisedRepresentationLearningTask:
    def __init__(
        self,
        *, 
        model: nn.Module, 
        use_gpu: bool = True,
        init_log_: bool = True, 
    ):
        self.use_gpu = use_gpu 
        self.device = auto_select_gpu(use_gpu=use_gpu)
        self.model = model.to(self.device)
        self.model.device = self.device 

        if init_log_:
            init_log() 
            
    def train_and_eval(
        self,
        *, 
        dataset: dict[str, Any],
        train_epoch: Callable,
        eval_epoch: Callable, 
        eval_epoch_interval: int = 1, 
        optimizer_type: str,
        optimizer_param: dict[str, Any],  
        max_epochs: int,
    ):
        optimizer = create_optimizer(type=optimizer_type, param=optimizer_param, model=self.model)

        epoch_val_acc_dict: dict[int, float] = dict() 
        epoch_test_acc_dict: dict[int, float] = dict() 

        dataset = to_device(dataset)
        
        for epoch in range(1, max_epochs + 1):
            self.model.train() 
            
            train_res = train_epoch(model=self.model, dataset=dataset)
            
            loss = train_res['loss']
                
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            
            log_info(f"[Train] Epoch: {epoch}, Loss: {float(loss):.4f}")

            if epoch % eval_epoch_interval == 0:
                self.model.eval()
                
                with torch.no_grad():
                    eval_res = eval_epoch(model=self.model, dataset=dataset)
                    
                val_acc = eval_res['val_acc']
                test_acc = eval_res['test_acc']
                    
                epoch_val_acc_dict[epoch] = val_acc 
                epoch_test_acc_dict[epoch] = test_acc 
                    
                sort_res = sorted(epoch_val_acc_dict.items(), key=lambda x: (-x[1], x[0]))
                best_val_acc_epoch, best_val_acc = sort_res[0]
                
                sort_res = sorted(epoch_test_acc_dict.items(), key=lambda x: (-x[1], x[0]))
                best_test_acc_epoch, best_test_acc = sort_res[0]
                    
                log_info(f"[Val] Epoch: {epoch}, Val Acc: {val_acc:.4f} Best Val Acc: {best_val_acc:.4f} (in Epoch {best_val_acc_epoch})")
                log_info(f"[Test] Epoch: {epoch}, Test Acc: {test_acc:.4f} Best Test Acc: {best_test_acc:.4f} (in Epoch {best_test_acc_epoch})")
