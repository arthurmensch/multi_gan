import copy
from collections import defaultdict
from itertools import chain

import torch
from torch import nn
from torch._six import container_abcs
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
        # Copied from pytorch code

        # Save ids instead of Tensors
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [id(p) for p in group['params']]
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use ids as keys
        packed_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.extra_state.items()}
        extra = {
            'state': packed_state,
            'param_groups': param_groups,
        }
        return dict(optimizer=self.optimizer.state_dict(), extrapolated=self.extrapolated, extra=extra)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.extrapolated = state_dict['extrapolated']

        # Copied from pytorch code

        # deepcopy, to be consistent with module API
        state_dict = state_dict['extra']
        state_dict = copy.deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        self.extra_state = state

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