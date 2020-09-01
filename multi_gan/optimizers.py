import copy
from collections import defaultdict
from itertools import chain

import torch
from torch import nn
from torch._six import container_abcs
from torch.optim.optimizer import Optimizer, required


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


class SignSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0, use_l1=False,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, use_l1=use_l1)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                alpha = - group['lr']
                if group['use_l1']:
                    alpha *= torch.abs(d_p).sum()
                d_p = torch.sign(d_p)
                p.add_(d_p, alpha=alpha)

        return loss
