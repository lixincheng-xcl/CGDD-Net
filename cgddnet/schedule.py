"""Linear warmup followed by cosine cycles with warm restarts."""
import math


class WarmupCosine:
    """LR multiplier indexed by optimizer update; checkpoint via LambdaLR.

    A restart resets learning rate only, not Adam's moment estimates.
    Warmup is applied once at the beginning, not after every restart.
    """
    def __init__(self, warmup_steps, first_cycle_steps, cycle_mult=2,
                 start_factor=0.1, min_factor=0.0):
        if warmup_steps < 0 or first_cycle_steps < 1 or cycle_mult < 1:
            raise ValueError("Invalid warmup/cycle lengths")
        if not 0 < start_factor <= 1 or not 0 <= min_factor <= 1:
            raise ValueError("Invalid learning-rate factors")
        self.warmup_steps = int(warmup_steps)
        self.first_cycle_steps = int(first_cycle_steps)
        self.cycle_mult = int(cycle_mult)
        self.start_factor, self.min_factor = start_factor, min_factor

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.start_factor + (1-self.start_factor)*step/self.warmup_steps
        elapsed, length = step-self.warmup_steps, self.first_cycle_steps
        if self.cycle_mult == 1:
            elapsed %= length
        else:
            while elapsed >= length:
                elapsed -= length
                length *= self.cycle_mult
        return self.min_factor + (1-self.min_factor)*(1+math.cos(math.pi*elapsed/length))/2
