"""Eight LIF neurons with hard forward spikes and the fixed surrogate gradient."""
import math
import torch


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a):
        ctx.save_for_backward(a)
        return (a>=0).to(a.dtype)
    @staticmethod
    def backward(ctx,gradient):
        a,=ctx.saved_tensors
        return gradient/(1+5*a.abs()).square()


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W=torch.nn.Parameter(torch.empty(8,48))
        self.A=torch.nn.Parameter(torch.empty(10,8))
        self.b=torch.nn.Parameter(torch.zeros(10))
        torch.nn.init.uniform_(self.W,-1/math.sqrt(48),1/math.sqrt(48))
        torch.nn.init.uniform_(self.A,-1/math.sqrt(8),1/math.sqrt(8))

    def forward(self,spikes,return_trace=False):
        if spikes.ndim!=3 or spikes.shape[1:]!=(12,48):raise ValueError('Expected N×12×48')
        current=spikes@self.W.T
        v=torch.zeros_like(current[:,0]);outputs=[];membranes=[]
        for t in range(12):
            u=.5*v+current[:,t]
            z=Spike.apply(u-1)
            v=(1-z.detach())*u
            outputs.append(z);membranes.append(v)
        z=torch.stack(outputs,dim=1)
        features=z.sum(dim=1)/12
        logits=features@self.A.T+self.b
        if return_trace:return logits,z,current,torch.stack(membranes,dim=1)
        return logits
