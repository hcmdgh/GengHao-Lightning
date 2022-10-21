from .imports import * 
from .device import * 
from .dataloader import *
from .optimizer import * 
from .metric import * 

from basic_util import * 

__all__ = [
    'MultiClassClassificationTask',
]


class MultiClassClassificationTask:
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

        if init_log_:
            init_log() 
            
    def convert_batch(self, 
                      batch: dict[str, Any]) -> dict[str, Any]:
        """
        将NumPy的Batch转换为Tensor，并移入GPU。
        """
        converted_batch = dict() 
        
        for k, v in batch.items():
            if isinstance(v, ndarray):
                if v.dtype != object:
                    v = torch.from_numpy(v).to(self.device)
            elif isinstance(v, Tensor):
                v = v.to(self.device)
            else:
                pass 
            
            converted_batch[k] = v
            
        return converted_batch             

    def train_and_eval(
        self,
        *, 
        train_dataloader: AbstractDataLoader,
        val_dataloader: Optional[AbstractDataLoader] = None,
        test_dataloader: Optional[AbstractDataLoader] = None,
        train_step: Callable,
        val_step: Optional[Callable] = None, 
        test_step: Optional[Callable] = None, 
        eval_epoch_interval: int = 1, 
        optimizer_type: str,
        optimizer_param: dict[str, Any],  
        max_epochs: int,
        tqdm_step: bool = True,
    ):
        optimizer = create_optimizer(type=optimizer_type, param=optimizer_param, model=self.model)

        epoch_val_acc_dict: dict[int, float] = dict() 
        epoch_test_acc_dict: dict[int, float] = dict() 
        
        for epoch in range(1, max_epochs + 1):
            self.model.train() 
            
            num_steps = train_dataloader.num_steps()
            loss_list = [] 
            
            for step, batch in enumerate(tqdm(train_dataloader, total=num_steps, desc='Training', disable=not tqdm_step or num_steps <= 5)):
                batch = self.convert_batch(batch=batch)
                
                train_res = train_step(model=self.model, batch=batch)
                pred = train_res['pred']
                target = train_res['target']
                loss = F.cross_entropy(input=pred, target=target)
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
                
                loss_list.append(float(loss))
                
            log_info(f"[Train] Epoch: {epoch}, Loss: {np.mean(loss_list):.4f}")

            if epoch % eval_epoch_interval == 0:
                self.model.eval()
                
                if val_step is not None and val_dataloader is not None:
                    num_steps = val_dataloader.num_steps()
                    
                    step_pred_list: list[FloatArray] = []
                    step_target_list: list[IntArray] = []
                    
                    for step, batch in enumerate(tqdm(val_dataloader, total=num_steps, desc='Validating', disable=not tqdm_step or num_steps <= 5)):
                        batch = self.convert_batch(batch=batch)
                        
                        with torch.no_grad():
                            val_res = val_step(model=self.model, batch=batch)
                        
                        pred = val_res['pred']
                        target = val_res['target']
                        
                        if isinstance(pred, Tensor):
                            pred = pred.detach().cpu().numpy() 
                        if isinstance(target, Tensor):
                            target = target.detach().cpu().numpy() 
                        
                        step_pred_list.append(pred)
                        step_target_list.append(target)
                        
                    full_pred = np.concatenate(step_pred_list, axis=0)
                    full_target = np.concatenate(step_target_list, axis=0)
                    
                    val_acc = calc_acc(pred=full_pred, target=full_target)
                    epoch_val_acc_dict[epoch] = val_acc 
                    
                    sort_res = sorted(epoch_val_acc_dict.items(), key=lambda x: (-x[1], x[0]))
                    best_val_acc_epoch, best_val_acc = sort_res[0]
                    
                    log_info(f"[Val] Epoch: {epoch}, Val Acc: {val_acc:.4f} Best Val Acc: {best_val_acc:.4f} (in Epoch {best_val_acc_epoch})")
                            
                if test_step is not None and test_dataloader is not None:
                    num_steps = test_dataloader.num_steps()
                    
                    step_pred_list: list[FloatArray] = []
                    step_target_list: list[IntArray] = []
                    
                    for step, batch in enumerate(tqdm(test_dataloader, total=num_steps, desc='Testing', disable=not tqdm_step or num_steps <= 5)):
                        batch = self.convert_batch(batch=batch)
                        
                        with torch.no_grad():
                            test_res = test_step(model=self.model, batch=batch)
                        
                        pred = test_res['pred']
                        target = test_res['target']
                        
                        if isinstance(pred, Tensor):
                            pred = pred.detach().cpu().numpy() 
                        if isinstance(target, Tensor):
                            target = target.detach().cpu().numpy() 
                        
                        step_pred_list.append(pred)
                        step_target_list.append(target)
                        
                    full_pred = np.concatenate(step_pred_list, axis=0)
                    full_target = np.concatenate(step_target_list, axis=0)
                    
                    test_acc = calc_acc(pred=full_pred, target=full_target)
                    epoch_test_acc_dict[epoch] = test_acc 
                    
                    sort_res = sorted(epoch_test_acc_dict.items(), key=lambda x: (-x[1], x[0]))
                    best_test_acc_epoch, best_test_acc = sort_res[0]
                    
                    log_info(f"[Test] Epoch: {epoch}, Test Acc: {test_acc:.4f} Best Test Acc: {best_test_acc:.4f} (in Epoch {best_test_acc_epoch})")
