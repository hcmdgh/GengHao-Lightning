from .imports import * 

__all__ = [
    'AbstractDataLoader',
    'DataLoader',
    'SingleBatchDataLoader',
]


class AbstractDataLoader:
    def __init__(self):
        pass
    
    def num_steps(self) -> int:
        raise NotImplementedError
    
    def __iter__(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


class DataLoader(AbstractDataLoader):
    def __init__(self,
                 dataset: dict[str, ndarray],
                 batch_size: int,
                 shuffle: bool):
        super().__init__()
        
        assert dataset 
        self.dataset = dataset 
        self.batch_size = batch_size 
        self.shuffle = shuffle 
        
        self.N = -1 
        
        for k, v in dataset.items():
            assert isinstance(v, ndarray)
            
            if self.N == -1:
                self.N = len(v)
            else:
                assert self.N == len(v)

    def num_steps(self) -> int:
        assert self.batch_size > 0
        
        return math.ceil(self.N / self.batch_size) 
                
    def __iter__(self) -> Iterator[dict[str, ndarray]]:
        if self.shuffle:
            perm = np.random.permutation(self.N)
        else:
            perm = np.arange(self.N)
            
        assert self.batch_size > 0 
            
        for i in range(0, self.N, self.batch_size):
            batch_dict = dict() 
            
            for k, v in self.dataset.items():
                v = v[perm[i : i + self.batch_size]]
                batch_dict[k] = v 
                
            yield batch_dict


class SingleBatchDataLoader(AbstractDataLoader):
    def __init__(self,
                 dataset: dict[str, Any]):
        super().__init__()
                
        assert dataset 
        self.dataset = dataset 
        
    def num_steps(self) -> int:
        return 1 
    
    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield self.dataset
