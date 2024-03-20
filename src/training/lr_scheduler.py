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
        super().get_lr(n_steps=n_steps)
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

class exponential_with_warm_up_scheduler(lr_scheduler):

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
        k = 0.1
        decay_ratio = math.exp(-k * (n_steps - self.config.lr_warmup_steps))
        assert 0 <= decay_ratio <= 1
        return self.config.lr * decay_ratio
    
class linear_with_warm_up_scheduler(lr_scheduler):
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
        decay_ratio = 1 - ((n_steps - self.config.lr_warmup_steps )/ (lr_decay_iters - self.config.lr_warmup_steps))
        assert 0 <= decay_ratio <= 1
        return self.config.lr * decay_ratio

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
        drop_rate = 0.8
        steps_drop = 100.0 # every steps_drop the lr drops
        decay_ratio = math.pow(drop_rate, math.floor((n_steps - self.config.lr_warmup_steps)/steps_drop))
        assert 0 <= decay_ratio <= 1
        return self.config.lr * decay_ratio

class time_based_with_warm_up_scheduler(lr_scheduler):

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
        decay = self.config.lr / (lr_decay_iters - self.config.lr_warmup_steps)
        decay_ratio = 1 / (1 + decay * (n_steps - self.config.lr_warmup_steps))
        assert 0 <= decay_ratio <= 1
        return self.config.lr * decay_ratio
# TODO: warmup restarts and cool downs ausprobieren beim besten scheduler und nach anderen Tricks gucken