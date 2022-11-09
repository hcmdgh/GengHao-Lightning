from ..imports import * 
from ..device import * 
from ..dataloader import * 
from ..evaluator import * 
from ..optimizer import * 

from basic_util import * 

__all__ = [
    'MiniBatchTrainer', 
]


class MiniBatchTrainer:
    def __init__(
        self,
        *, 
        model: nn.Module, 
        use_gpu: bool = True,
        do_init_log: bool = True, 
        use_wandb: bool = True,
        project_name: Optional[str] = None, 
        param_dict: Optional[dict[str, Any]] = None, 
    ):
        self.use_gpu = use_gpu 
        self.device = auto_select_gpu(use_gpu=use_gpu)
        self.model = model.to(self.device)
        
        # 为模型赋予device属性，便于在模型内部访问device
        self.model.device = self.device 

        if do_init_log:
            init_log() 
            
        self.use_wandb = use_wandb
        if use_wandb:
            assert project_name 
            
            wandb.init(project=project_name, config=param_dict)
            
    def train_and_eval(
        self,
        *, 
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        train_func: Callable,
        after_train_func: Optional[Callable] = None, 
        val_func: Optional[Callable] = None, 
        test_func: Optional[Callable] = None, 
        evaluator: BaseEvaluator,
        eval_interval: int = 1, 
        save_model_interval: int = -1, 
        optimizer_type: str,
        optimizer_param: dict[str, Any],  
        num_epochs: int,
        tqdm_step: bool = True, 
    ):
        device = get_device()
        
        optimizer = create_optimizer(type=optimizer_type, param=optimizer_param, model=self.model)

        for epoch in range(1, num_epochs + 1):
            self.model.train() 
            
            num_steps = len(train_dataloader)
            loss_list = [] 
            
            for step, batch in enumerate(train_dataloader, start=1):
                assert isinstance(batch, dict)
                batch = to_device(batch)
                
                train_result = train_func(epoch=epoch, model=self.model, **batch)
                    
                loss = evaluator.eval_train_step(epoch=epoch, step=step, num_steps=num_steps, **train_result)
                loss_list.append(float(loss))

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
            
            evaluator.eval_train_epoch(
                epoch = epoch, 
                loss = np.mean(loss_list), 
            )
            
            if after_train_func is not None:
                after_train_func(epoch=epoch, model=self.model)

            if epoch % eval_interval == 0:
                self.model.eval()
                
                if val_dataloader is not None and val_func is not None:
                    step_data_dict: dict[int, dict[str, Any]] = dict() 
                    
                    for step, batch in enumerate(tqdm(val_dataloader, desc='Validating', disable=not tqdm_step), start=1):
                        assert isinstance(batch, dict)
                        batch = to_device(batch)
                        
                        step_data = val_func(epoch=epoch, step=step, model=self.model, **batch)
                        step_data_dict[step] = to_numpy(step_data)          
                
                    evaluator.eval_val_steps_in_one_epoch(
                        epoch = epoch, 
                        step_data_dict = step_data_dict, 
                    )
                    
                if test_dataloader is not None and test_func is not None:
                    step_data_dict: dict[int, dict[str, Any]] = dict() 
                    
                    for step, batch in enumerate(tqdm(test_dataloader, desc='Testing', disable=not tqdm_step), start=1):
                        assert isinstance(batch, dict)
                        batch = to_device(batch)
                        
                        step_data = test_func(epoch=epoch, step=step, model=self.model, **batch)
                        step_data_dict[step] = to_numpy(step_data)        
                
                    evaluator.eval_test_steps_in_one_epoch(
                        epoch = epoch, 
                        step_data_dict = step_data_dict, 
                    )

            if save_model_interval > 0 and epoch % save_model_interval == 0:
                os.makedirs('./saved_model', exist_ok=True)
                
                torch.save(self.model.state_dict(), f"./saved_model/model_state_epoch_{epoch}.pt")

        if self.use_wandb:
            evaluator.summary() 
