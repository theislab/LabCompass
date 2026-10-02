import torch

__all__ = ["SDE"]


class SDE(torch.nn.Module):
    def __init__(
        self,
        vf_fn,
        sde_type = "ito",
        noise_type = "diagonal",
        k=1.0,
    ):
        self.vf_fn = vf_fn
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.k = k
    
    def f(self, t, xt):
        vt = self.vf_fn(t, xt)
        return (1 + self.k*t)*vt -self.k* xt

    def g(self, t, xt):
        t = torch.ones_like(xt) * t
        return torch.sqrt(self.k*2*(1 - t))