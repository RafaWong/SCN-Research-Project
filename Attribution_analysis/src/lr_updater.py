from math import cos, pi

class CosineLrUpdater():
    """LR Scheduler in runnver_base

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """
    
    def __init__(self,
                 optimizer,
                 periods,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None):
        
        assert (min_lr is None) ^ (min_lr_ratio is None)
       
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'
        self.optimizer = optimizer
        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

    def _set_lr(self, lr_groups):
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, epoch, iter, base_lr):
        if self.by_epoch:
            progress = epoch
        else:
            progress = iter

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr[idx], alpha, current_weight)


    def get_regular_lr(self, epoch, iter):
        return [self.get_lr(epoch, iter, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in self.optimizer.param_groups
        ]

    def before_train_epoch(self, len_train_data, epoch, iter):
        if not self.by_epoch:
            return
        if self.warmup_by_epoch:
            epoch_len = len_train_data
            self.warmup_iters = self.warmup_epochs * epoch_len

        self.regular_lr = self.get_regular_lr(epoch, iter)
        self._set_lr(self.regular_lr)

    def before_train_iter(self, epoch, iter):
        cur_iter = iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(epoch, iter)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(warmup_lr)

def get_position_from_periods(iteration, cumulative_periods):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_periods = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 3.

    Args:
        iteration (int): Current iteration.
        cumulative_periods (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out
