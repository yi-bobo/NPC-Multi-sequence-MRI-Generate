
import inspect
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

def create_optimizer(optimizer_name: str, model_params, **kwargs):
    """
        optimizer_name (str): 选择的优化器名称（如 'Adam', 'SGD' 等）
        model_params: 模型的参数 (通常为 model.parameters())
        **kwargs: 仅为指定优化器提供所需的参数，函数自动过滤其余无效参数
    """
    # 优化器映射（可根据你项目结构替换模块路径）
    optimizer_classes = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
        "Adadelta": optim.Adadelta,
        "Adamax": optim.Adamax,
        "NAdam": optim.NAdam,
        "Rprop": optim.Rprop,
        "ASGD": optim.ASGD,
        "LBFGS": optim.LBFGS,
        "SparseAdam": optim.SparseAdam,
        "RAdam": optim.RAdam,  # 如果你引入了额外扩展优化器
    }

    if optimizer_name not in optimizer_classes:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 获取优化器类
    optimizer_class = optimizer_classes[optimizer_name]

    # 获取构造函数支持的参数
    valid_args = inspect.signature(optimizer_class).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

    # 实例化优化器
    return optimizer_class(model_params, **filtered_kwargs)

def create_scheduler(scheduler_name: str, optimizer, **kwargs):
    """
        scheduler_name (str): 调度器名称（如 'StepLR', 'CosineAnnealingLR' 等）
        optimizer: 优化器实例
        **kwargs: 当前调度器支持的参数（如 step_size, gamma 等）
    """
    scheduler_classes = {
        "StepLR": lr_scheduler.StepLR,
        "MultiStepLR": lr_scheduler.MultiStepLR,
        "ExponentialLR": lr_scheduler.ExponentialLR,
        "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
        "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
        "CyclicLR": lr_scheduler.CyclicLR,
        "OneCycleLR": lr_scheduler.OneCycleLR,
        "LambdaLR": lr_scheduler.LambdaLR,
        "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
        "GradualWarmupScheduler": GradualWarmupScheduler
    }

    if scheduler_name not in scheduler_classes:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    scheduler_class = scheduler_classes[scheduler_name]

    # 获取调度器构造函数参数，并过滤
    valid_args = inspect.signature(scheduler_class).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

    return scheduler_class(optimizer, **filtered_kwargs)

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
        
