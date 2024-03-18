from .config import TrainingConfig
import math

class lr_scheduler:
    
    def __init__(self, config: TrainingConfig):
        pass

    def get_lr(self, n_steps):
        pass

class cosine_with_warm_up_scheduler(lr_scheduler):

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.config = config

    def get_lr(self, n_steps):
        super().get_lr(n_steps=n_steps)
        lr_decay_iters = self.config.n_steps
        # 1) linear warmup for warmup_iters steps
        if n_steps < self.config.lr_warmup_steps:
            return self.config.lr * n_steps / self.config.lr_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if n_steps > lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (n_steps - self.config.lr_warmup_steps) / (lr_decay_iters - self.config.lr_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)

class polynomial_with_warm_up_scheduler(lr_scheduler):

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.config = config

    def get_lr(self, n_steps):
        lr_decay_iters = self.config.n_steps
        # 1) linear warmup for warmup_iters steps
        if n_steps < self.config.lr_warmup_steps:
            return self.config.lr * n_steps / self.config.lr_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if n_steps > lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = 1 - ((n_steps - self.config.lr_warmup_steps )/ (lr_decay_iters - self.config.lr_warmup_steps)) ** 2
        assert 0 <= decay_ratio <= 1
        return self.config.lr * decay_ratio
    
# TODO: adjust
class exponential_with_warm_up_scheduler(lr_scheduler):

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.config = config

    def get_lr(self, n_steps):
        lr_decay_iters = self.config.n_steps
        # 1) linear warmup for warmup_iters steps
        if n_steps < self.config.lr_warmup_steps:
            return self.config.lr * n_steps / self.config.lr_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if n_steps > lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (n_steps - self.config.lr_warmup_steps) / (lr_decay_iters - self.config.lr_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)
    
# TODO: linear decay
class linear_with_warm_up_scheduler(lr_scheduler):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.config = config

    def get_lr(self, n_steps):
        lr_decay_iters = self.config.n_steps
        # 1) linear warmup for warmup_iters steps
        if n_steps < self.config.lr_warmup_steps:
            return self.config.lr * n_steps / self.config.lr_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if n_steps > lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (n_steps - self.config.lr_warmup_steps) / (lr_decay_iters - self.config.lr_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)

# TODO: step based
class step_based_with_warm_up_scheduler(lr_scheduler):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.config = config

    def get_lr(self, n_steps):
        lr_decay_iters = self.config.n_steps
        # 1) linear warmup for warmup_iters steps
        if n_steps < self.config.lr_warmup_steps:
            return self.config.lr * n_steps / self.config.lr_warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if n_steps > lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (n_steps - self.config.lr_warmup_steps) / (lr_decay_iters - self.config.lr_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.lr - self.config.min_lr)
