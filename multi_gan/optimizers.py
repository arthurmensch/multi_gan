import copy
from collections import defaultdict

from torch import nn
from torch.optim.optimizer import Optimizer


class ExtraOptimizer:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups

        self.extra_state = defaultdict(dict)
        for group in self.param_groups:
            for p in group['params']:
                self.extra_state[p]['backup'] = p.data.clone()
        self.extrapolated = False

    def share_memory(self):
        self.optimizer.share_memory()
        for group in self.param_groups:
            for p in group['params']:
                self.extra_state[p]['backup'].share_memory_()

    def _save_backup(self):
        for group in self.param_groups:
            for p in group['params']:
                self.extra_state[p]['backup'].copy_(p.data)

    def deextrapolate(self):
        if self.extrapolated:
            for group in self.param_groups:
                for p in group['params']:
                    p.data.copy_(self.extra_state[p]['backup'])

    def state_dict(self):
        for group in self.param_groups:
            for p in group['params']:
                for key in self.extra_state[p]:
                    self.optimizer.state[p][key] = self.extra_state[p][key]
        state_dict = self.optimizer.state_dict()
        for group in self.param_groups:
            for p in group['params']:
                for key in self.extra_state[p]:
                    self.optimizer.state[p].pop(key)
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                for key in self.extra_state[p]:
                    self.extra_state[p][key] = self.optimizer.state[p].pop(key)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None, extrapolate=False):
        update = not extrapolate
        if update:
            self.deextrapolate()
        loss = self.optimizer.step(closure)
        if update:
            self._save_backup()
        self.extrapolated = extrapolate
        return loss


class ParamAverager(nn.Module):
    def __init__(self, module, mode='uniform', beta=0.9):
        super().__init__()
        self.module = module
        self.average = copy.deepcopy(module)
        self.mode = mode
        self.n_step = 0
        self.beta = beta

    def step(self):
        self.n_step += 1
        for avg, p in zip(self.average.parameters(), self.module.parameters()):
            if self.mode == 'uniform':
                avg.data *= (1 - 1 / self.n_step)
                avg.data += p.data / self.n_step
            else:
                avg.data *= self.beta
                avg.data += p.data * (1 - self.beta)