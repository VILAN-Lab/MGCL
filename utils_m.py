import torch

def repeat_tensors(n, x):
    if torch.is_tensor(x):
        x = x.unsqueeze(1)
        x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
        x = x.reshape(x.shape[0]*n, *x.shape[2:])
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x