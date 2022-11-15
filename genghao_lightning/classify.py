from .imports import * 
from .device import * 
from .metric import * 

from basic_util import * 

__all__ = [
    'linear_classify',
]


def linear_classify(*,
                    train_feat: FloatTensor,
                    train_label: IntTensor,
                    val_feat: FloatTensor,
                    val_label: IntTensor,
                    test_feat: FloatTensor,
                    test_label: IntTensor,
                    use_gpu: bool = True, 
                    lr: float = 0.001,
                    num_epochs: int = 500,
                    use_tqdm: bool = True) -> dict[str, Any]:
    device = auto_select_gpu(use_gpu=use_gpu)

    train_feat = train_feat.detach().to(device)
    train_label = train_label.detach().to(device)
    val_feat = val_feat.detach().to(device)
    val_label = val_label.detach().to(device)
    test_feat = test_feat.detach().to(device)
    test_label = test_label.detach().to(device)

    feat_dim = train_feat.shape[-1]

    total_label = torch.concat([train_label, val_label, test_label])
    num_classes = len(total_label.unique())

    model = nn.Linear(feat_dim, num_classes)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_to_val_f1_micro: dict[int, float] = dict() 
    epoch_to_val_f1_macro: dict[int, float] = dict() 
    epoch_to_test_f1_micro: dict[int, float] = dict()
    epoch_to_test_f1_macro: dict[int, float] = dict()

    for epoch in tqdm(range(1, num_epochs + 1), disable=not use_tqdm, desc='Linear Classify', unit='epoch'):
        model.train() 
        
        pred = model(train_feat)
        
        loss = F.cross_entropy(input=pred, target=train_label)
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
        model.eval() 
        
        with torch.no_grad():
            val_pred = model(val_feat)
            test_pred = model(test_feat)

        val_f1_micro = calc_f1_micro(pred=val_pred, target=val_label)
        val_f1_macro = calc_f1_macro(pred=val_pred, target=val_label)
        test_f1_micro = calc_f1_micro(pred=test_pred, target=test_label)
        test_f1_macro = calc_f1_macro(pred=test_pred, target=test_label)
        
        epoch_to_val_f1_micro[epoch] = val_f1_micro
        epoch_to_val_f1_macro[epoch] = val_f1_macro
        epoch_to_test_f1_micro[epoch] = test_f1_micro
        epoch_to_test_f1_macro[epoch] = test_f1_macro

    best_val_f1_micro_epoch, best_val_f1_micro = max(epoch_to_val_f1_micro.items(), key=lambda x: (x[1], -x[0]))
    best_val_f1_macro_epoch, best_val_f1_macro = max(epoch_to_val_f1_macro.items(), key=lambda x: (x[1], -x[0]))
    best_test_f1_micro_epoch, best_test_f1_micro = max(epoch_to_test_f1_micro.items(), key=lambda x: (x[1], -x[0]))
    best_test_f1_macro_epoch, best_test_f1_macro = max(epoch_to_test_f1_macro.items(), key=lambda x: (x[1], -x[0]))

    if best_val_f1_micro_epoch >= num_epochs - 20 \
    or best_test_f1_micro_epoch >= num_epochs - 20:
        log_warning(f"Linear模型尚未完全收敛！")

    # 释放显存
    del train_feat
    del train_label
    del val_feat
    del val_label
    del test_feat
    del test_label
    del model 
        
    return dict(
        best_val_f1_micro_epoch = best_val_f1_micro_epoch,
        best_val_f1_micro = best_val_f1_micro,
        best_val_f1_macro_epoch = best_val_f1_macro_epoch,
        best_val_f1_macro = best_val_f1_macro,
        best_test_f1_micro_epoch = best_test_f1_micro_epoch,
        best_test_f1_micro = best_test_f1_micro,
        best_test_f1_macro_epoch = best_test_f1_macro_epoch,
        best_test_f1_macro = best_test_f1_macro, 
    )
