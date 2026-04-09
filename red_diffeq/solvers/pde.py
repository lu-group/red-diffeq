import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FWIForward(nn.Module):

    def __init__(self, ctx, device, sample_temporal=1, sample_spatial=1.0, normalize=True, v_denorm_func=None, s_norm_func=None):
        super(FWIForward, self).__init__()
        self.device = device
        self.normalize = normalize
        if normalize:
            self.v_denorm_func = v_denorm_func
            self.s_norm_func = s_norm_func
        self.sample_temporal = sample_temporal
        if 'sx' not in ctx.keys():
            ctx['sx'] = np.linspace(0, ctx['n_grid'] - 1, num=ctx['ns']) * ctx['dx']
        else:
            ctx['sx'] = np.array(ctx['sx']) * ctx['dx']
        if 'gx' not in ctx.keys():
            ctx['gx'] = np.linspace(0, ctx['n_grid'] - 1, num=int(sample_spatial * ctx['ng'])) * ctx['dx']
        else:
            ctx['gx'] = np.array(ctx['gx']) * ctx['dx']
        self.ctx = ctx

    def ricker(self, f, dt, nt):
        nw = 2.2 / f / dt
        nw = 2 * np.floor(nw / 2) + 1
        nc = np.floor(nw / 2)
        k = np.arange(nw)
        alpha = (nc - k) * f * dt * np.pi
        beta = alpha ** 2
        w0 = (1 - beta * 2) * np.exp(-beta)
        w = np.zeros(nt)
        w[:len(w0)] = w0
        return w

    def get_Abc(self, vp, nbc, dx):
        dimrange = 1.0 * torch.unsqueeze(torch.arange(nbc, device=self.device), dim=-1)
        damp = torch.zeros_like(vp, device=self.device, requires_grad=False)
        velmin, _ = torch.min(vp.view(vp.shape[0], -1), dim=-1, keepdim=False)
        a = (nbc - 1) * dx
        kappa = 3.0 * velmin * np.log(10000000.0) / (2.0 * a)
        kappa = torch.unsqueeze(kappa, dim=0)
        kappa = torch.repeat_interleave(kappa, nbc, dim=0)
        damp1d = kappa * (dimrange * dx / a) ** 2
        damp1d = damp1d.permute(1, 0).unsqueeze(1)
        damp[:, :, :nbc, :] = torch.repeat_interleave(torch.flip(damp1d, dims=[-1]).unsqueeze(-1), vp.shape[-1], dim=-1)
        damp[:, :, -nbc:, :] = torch.repeat_interleave(damp1d.unsqueeze(-1), vp.shape[-1], dim=-1)
        damp[:, :, :, :nbc] = torch.repeat_interleave(torch.flip(damp1d, dims=[-1]).unsqueeze(-2), vp.shape[-2], dim=-2)
        damp[:, :, :, -nbc:] = torch.repeat_interleave(damp1d.unsqueeze(-2), vp.shape[-2], dim=-2)
        return damp

    def adj_sr(self, sx, sz, gx, gz, dx, nbc):
        isx = np.around(sx / dx) + nbc
        isz = np.around(sz / dx) + nbc
        igx = np.around(gx / dx) + nbc
        igz = np.around(gz / dx) + nbc
        return (isx.astype('int'), int(isz), igx.astype('int'), int(igz))

    def FWM(self, v, nbc, dx, nt, dt, f, sx, sz, gx, gz, **kwargs):
        src = self.ricker(f, dt, nt)
        alpha = (v * dt / dx) ** 2
        abc = self.get_Abc(v, nbc, dx)
        kappa = abc * dt
        c1 = -2.5
        c2 = 4.0 / 3.0
        c3 = -1.0 / 12.0
        temp1 = 2 + 2 * c1 * alpha - kappa
        temp2 = 1 - kappa
        beta_dt = (v * dt) ** 2
        ns = len(sx)
        isx, isz, igx, igz = self.adj_sr(sx, sz, gx, gz, dx, nbc)
        seis = []
        p1 = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=self.device, requires_grad=True)
        p0 = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=self.device, requires_grad=True)
        p = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=self.device, requires_grad=True)
        for i in range(nt):
            p = temp1 * p1 - temp2 * p0 + alpha * (c2 * (torch.roll(p1, 1, dims=-2) + torch.roll(p1, -1, dims=-2) + torch.roll(p1, 1, dims=-1) + torch.roll(p1, -1, dims=-1)) + c3 * (torch.roll(p1, 2, dims=-2) + torch.roll(p1, -2, dims=-2) + torch.roll(p1, 2, dims=-1) + torch.roll(p1, -2, dims=-1)))
            for loc in range(ns):
                p[:, loc, isz, isx[loc]] = p[:, loc, isz, isx[loc]] + beta_dt[:, 0, isz, isx[loc]] * src[i]
            if i % self.sample_temporal == 0:
                seis.append(torch.unsqueeze(p[:, :, [igz] * len(igx), igx], dim=2))
            p0 = p1
            p1 = p
        return torch.cat(seis, dim=2)

    def forward(self, v):
        if self.normalize:
            v = self.v_denorm_func(v)
        v_pad = F.pad(v, (self.ctx['nbc'],) * 4, mode='replicate')
        s = self.FWM(v_pad, **self.ctx)
        return self.s_norm_func(s) if self.normalize else s