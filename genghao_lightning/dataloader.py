from .imports import * 

__all__ = [
    # 'DataLoader',
]


class DataLoader:
    def __init__(self,
                 dataset: dict[str, ndarray],
                 batch_size: int,
                 shuffle: bool):
        raise DeprecationWarning
        assert dataset and batch_size >= 1 
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
