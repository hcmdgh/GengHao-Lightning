from .imports import * 
from .device import * 
from .dataloader import * 
from .evaluator import * 
from .optimizer import * 

from basic_util import * 

__all__ = [
    'FullBatchTrainer', 
]


class FullBatchTrainer:
    def __init__(
        self,
        *, 
        model: nn.Module, 
        use_gpu: bool = True,
        do_init_log: bool = True, 
        use_wandb: bool = True,
        project_name: Optional[str] = '', 
    ):
        self.use_gpu = use_gpu 
        self.device = auto_select_gpu(use_gpu=use_gpu)
        self.model = model.to(self.device)
        
        # 模型新增device属性，便于在模型内部访问device
        self.model.device = self.device 

        if do_init_log:
            init_log() 
            
        if use_wandb:
            assert project_name 
            
            wandb.init(project=project_name)
            
    def train_and_eval(
        self,
        *, 
        dataset: dict[str, Any],
        train_func: Callable,
        val_func: Optional[Callable] = None, 
        test_func: Optional[Callable] = None, 
        evaluator: BaseEvaluator,
        eval_interval: int = 1, 
        save_model_interval: int = -1, 
        optimizer_type: str,
        optimizer_param: dict[str, Any],  
        num_epochs: int,
    ):
        optimizer = create_optimizer(type=optimizer_type, param=optimizer_param, model=self.model)

        dataset = to_device(dataset)
        
        for epoch in range(1, num_epochs + 1):
            self.model.train() 
            
            train_result = train_func(epoch=epoch, model=self.model, **dataset)
                
            loss = evaluator.eval_train_epoch(epoch=epoch, **train_result)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 

            if epoch % eval_interval == 0:
                self.model.eval()
                
                if val_func is not None:
                    with torch.no_grad():
                        val_result = val_func(epoch=epoch, model=self.model, **dataset)
                        
                    evaluator.eval_val_epoch(epoch=epoch, **val_result)
                    
                if test_func is not None:
                    with torch.no_grad():
                        test_result = test_func(epoch=epoch, model=self.model, **dataset)
                        
                    evaluator.eval_test_epoch(epoch=epoch, **test_result)

            if save_model_interval > 0 and epoch % save_model_interval == 0:
                os.makedirs('./saved_model', exist_ok=True)
                
                torch.save(self.model.state_dict(), f"./saved_model/model_state_epoch_{epoch}.pt")
