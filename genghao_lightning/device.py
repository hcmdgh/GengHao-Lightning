from .imports import * 

__all__ = [
    'auto_select_gpu', 
    'get_device', 
]

_device = None 


def get_device() -> torch.device:
    assert _device is not None 
    
    return _device 


def auto_select_gpu(use_gpu: bool = True) -> torch.device:
    global _device 

    # 只设置一次device
    if _device is not None:
        return _device

    if not use_gpu:
        _device = torch.device('cpu')
        return _device
    
    exe_res = os.popen('gpustat --json').read() 
    
    state_dict = json.loads(exe_res)
    
    gpu_infos = [] 
    
    for gpu_entry in state_dict['gpus']:
        gpu_id = int(gpu_entry['index'])
        used_mem = int(gpu_entry['memory.used'])

        gpu_infos.append((used_mem, gpu_id))
    
    gpu_infos.sort()
    
    _device = torch.device(f'cuda:{gpu_infos[0][1]}')
    
    return _device 
