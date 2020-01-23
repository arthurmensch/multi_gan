import torch
from torch.nn import functional as F, Parameter


def compute_gan_loss(true_logits, fake_logits, loss_type='gan'):
    """Return the generator loss"""
    if loss_type == 'gan':  # min log(1 - D(G(z))
        loss_G = F.logsigmoid(true_logits).mean() - F.softplus(fake_logits).mean()
        loss_D = - loss_G
    elif loss_type == 'ns-gan':
        loss_G = - F.logsigmoid(fake_logits).mean()
        loss_D = - F.logsigmoid(true_logits).mean() + F.softplus(fake_logits).mean()
    elif loss_type in ['wgan', 'wgan-gp']:
        loss_G = fake_logits.mean() - true_logits.mean()
        loss_D = - loss_G
    else:
        raise NotImplementedError()

    return {'G': loss_G, 'D': loss_D}


def compute_grad_penalty(net_D, true_data, fake_data):
    batch_size = true_data.shape[0]
    epsilon = true_data.new(batch_size, 1, 1, 1)
    epsilon = epsilon.uniform_()
    line_data = true_data * (1 - epsilon) + fake_data * (1 - epsilon)
    line_data = Parameter(line_data)
    line_pred = net_D(line_data).sum()
    grad, = torch.autograd.grad(line_pred, line_data, create_graph=True)
    grad = grad.view(batch_size, -1)
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    return ((grad_norm - 1) ** 2).mean()