from .imports import * 
from .device import * 
from .dataloader import *
from .optimizer import * 
from .metric import * 

from basic_util import * 

__all__ = [
    'Trainer',
]


class Trainer:
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
        train_dataloader: Union[DataLoader, NoBatchDataLoader],
        val_dataloader: Optional[Union[DataLoader, NoBatchDataLoader]] = None,
        test_dataloader: Optional[Union[DataLoader, NoBatchDataLoader]] = None,
        train_step: Callable,
        val_step: Optional[Callable] = None, 
        test_step: Optional[Callable] = None, 
        eval_epoch_interval: int = 1, 
        eval_metric, 
        optimizer: Union[dict[str, Any], optim.Optimizer], 
        max_epochs: int,
        tqdm_step: bool = True,
    ):
        if isinstance(optimizer, dict):
            optimizer = create_optimizer(param=optimizer, model=self.model)
        assert isinstance(optimizer, optim.Optimizer)
        
        val_metric = eval_metric(val_or_test='val')
        test_metric = eval_metric(val_or_test='test')
        
        for epoch in range(1, max_epochs + 1):
            self.model.train() 
            
            if isinstance(train_dataloader, DataLoader):
                num_steps = train_dataloader.num_steps()
                loss_list = [] 
                
                for step, batch in enumerate(tqdm(train_dataloader, total=num_steps, desc='Training', disable=not tqdm_step)):
                    batch = self.convert_batch(batch=batch)
                    loss = train_step(model=self.model, batch=batch)
                    
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step() 
                    
                    loss_list.append(float(loss))
                    
                log_info(f"[Train] Epoch: {epoch}, Loss: {np.mean(loss_list):.4f}")

            elif isinstance(train_dataloader, NoBatchDataLoader):
                batch = self.convert_batch(train_dataloader.dataset)
                loss = train_step(model=self.model, batch=batch)
                    
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
                
                log_info(f"[Train] Epoch: {epoch}, Loss: {float(loss):.4f}")
                
            else:
                raise TypeError
            
            if epoch % eval_epoch_interval == 0:
                self.model.eval()
                
                if val_step is not None and val_dataloader is not None:
                    
                    if isinstance(val_dataloader, DataLoader):
                        num_steps = val_dataloader.num_steps()
                        
                        for step, batch in enumerate(tqdm(val_dataloader, total=num_steps, desc='Validating', disable=not tqdm_step)):
                            batch = self.convert_batch(batch=batch)
                            
                            with torch.no_grad():
                                res = val_step(model=self.model, batch=batch)
                            
                            val_metric.record_step(
                                y_pred = res['y_pred'],
                                y_true = res['y_true'],
                            )
                            
                        val_metric.summary_all_step(epoch=epoch)
                            
                    elif isinstance(val_dataloader, NoBatchDataLoader):
                        batch = self.convert_batch(val_dataloader.dataset)
                        
                        with torch.no_grad():
                            res = val_step(model=self.model, batch=batch)
                            
                        val_metric.record_epoch(
                            epoch = epoch, 
                            y_pred = res['y_pred'],
                            y_true = res['y_true'],
                        )

                    else:
                        raise AssertionError
                        
                if test_step is not None and test_dataloader is not None:
                    
                    if isinstance(test_dataloader, DataLoader):
                        num_steps = test_dataloader.num_steps()
                        
                        for step, batch in enumerate(tqdm(test_dataloader, total=num_steps, desc='Testing', disable=not tqdm_step)):
                            batch = self.convert_batch(batch=batch)
                            
                            with torch.no_grad():
                                res = test_step(model=self.model, batch=batch)
                            
                            test_metric.record_step(
                                y_pred = res['y_pred'],
                                y_true = res['y_true'],
                            )
                            
                        test_metric.summary_all_step(epoch=epoch)
                            
                    elif isinstance(test_dataloader, NoBatchDataLoader):
                        batch = self.convert_batch(test_dataloader.dataset)
                        
                        with torch.no_grad():
                            res = test_step(model=self.model, batch=batch)
                            
                        test_metric.record_epoch(
                            epoch = epoch, 
                            y_pred = res['y_pred'],
                            y_true = res['y_true'],
                        )

                    else:
                        raise AssertionError
