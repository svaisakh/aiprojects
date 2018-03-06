import numpy as np
import torch.nn as nn

mu_law = lambda x, mu=255: np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
inv_mu_law = lambda x, mu=255: np.sign(x) * ((1 + mu) ** np.abs(x) -1) / mu

gaussian_filter = lambda x, buckets, std=1: softmax(-((buckets - np.expand_dims(x, -1)) ** 2) / (2 * std ** 2))

def softmax(x):
    x -= x.max(-1, keepdims=True)
    x = np.exp(x)
    return x / x.sum(-1, keepdims=True)

def one_hot(x, classes=None, dtype=np.float32):
    if classes is None: classes = x.max() + 1
    return np.eye(classes, dtype=dtype)[x]

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, 1, dilation * (kernel_size - 1), dilation, 1, bias)

    def forward(self, x):
        x = super().forward(x)
        return x[:, :, :-self.padding[0]] if self.padding[0] > 0 else x

def print_tree(output):
    g_fn = [output.grad_fn]
            
    while True:
        print(g_fn, '\n')
        new_g_fn = []
        for g in g_fn:
            if g is None: continue
            next_fn = g.next_functions
            for fn, _ in next_fn:
                new_g_fn.append(fn)
        g_fn = new_g_fn
        if len(g_fn) == 0: break

def show_zero_grads(module=None, percent_only=True):
    if module is None:
        if not percent_only:
            for c in model.children():
                for p in c.parameters():
                    if p.grad is None: continue
                    x = np.where(p.grad.data.cpu().numpy() == 0)
                    if len(x[0]) > 0: 
                        return "{} {} {}% {}".format(c, len(x[0]), int(100 * len(x[0]) / len(p.view(-1))), '%', p.size())
        else:
            z_count = 0
            count = 0
            for c in model.children():
                for p in c.parameters():
                    if p.grad is None: continue
                    x = np.where(p.grad.data.cpu().numpy() == 0)
                    z_count += len(x[0])
                    count += len(p.view(-1))
            return 100 * z_count / count
                    
    else:
        for p in module.parameters():
            x = np.where(p.grad.data.cpu().numpy() == 0)
            return 100 * len(x[0]) / len(p.view(-1))