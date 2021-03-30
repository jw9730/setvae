import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from .ops import get_mask, sample_mask, masked_fill, get_pairwise_distance
from .fiblat import sphere_lattice


class InitialSet(nn.Module):
    def __init__(self, args, dim_seed, n_mixtures, fixed_gmm, train_gmm, dim_out, max_outputs):
        super().__init__()
        self.dim_seed = dim_seed
        self.n_mixtures = n_mixtures
        self.dim_out = dim_out
        self.max_outputs = max_outputs
        self.n_mixtures = n_mixtures
        self.fixed_gmm = fixed_gmm
        self.train_gmm = train_gmm
        self.tau = 1.
        self.register = self.register_parameter if args.train_gmm else self.register_buffer

        if n_mixtures == 1:
            self.register('mu', nn.Parameter(torch.randn(1, 1, dim_seed)))  # [1, 1, Ds]
            self.register('logvar', nn.Parameter(torch.randn(1, 1, dim_seed)))  # [1, 1, Ds]
            nn.init.xavier_uniform_(self.mu)
            nn.init.xavier_uniform_(self.logvar)
        elif fixed_gmm:
            logits, mu, sig = self.get_mixture_parameters(n_mixtures, dim_seed)
            self.register('logits', nn.Parameter(logits))
            self.register('mu', nn.Parameter(mu))
            self.register('sig', nn.Parameter(sig))
        else:
            # torch.manual_seed(42)
            self.register('logits', nn.Parameter(torch.ones(n_mixtures,)))
            self.register('mu', nn.Parameter(torch.randn(n_mixtures, dim_seed)))
            self.register('sig', nn.Parameter(torch.randn(n_mixtures, dim_seed).abs() / math.sqrt(n_mixtures)))
        self.output = nn.Linear(dim_seed, dim_out)

    @staticmethod
    def get_mixture_parameters(n_mixtures, dim_seed, downscale=10., eps=1e-3):
        logits = torch.ones(n_mixtures,)  # [N,]
        mu = torch.tensor(sphere_lattice(dim_seed, n_mixtures))  # [N, D]
        # ensure separability of any pair of components
        tril = torch.ones(n_mixtures, n_mixtures).tril(diagonal=-1).bool()
        s = get_pairwise_distance(mu)[tril].min().item() / downscale  # this is empirical
        sig = torch.ones(n_mixtures, dim_seed) * (s + eps)  # [N, D]
        return logits, mu, sig

    def forward(self, output_sizes, hold_seed=None, hold_initial_set=False):
        """
        Sample from prior
        :param output_sizes: Tensor([B,])
        :param hold_seed
        :param hold_initial_set
        :return: Tensor([B, N, D])
        """
        bsize = output_sizes.shape[0]
        if hold_initial_set:  # [B, N]
            x_mask = get_mask(output_sizes, self.max_outputs)
        else:
            x_mask = sample_mask(output_sizes, self.max_outputs)

        if hold_seed is not None:  # [B, N, Ds]
            torch.random.manual_seed(hold_seed)
            eps = torch.randn([1, self.max_outputs, self.dim_seed]).to(x_mask.device).repeat(bsize, 1, 1)
        else:
            eps = torch.randn([bsize, self.max_outputs, self.dim_seed]).to(x_mask.device)

        if self.n_mixtures == 1:
            x = self.mu + torch.exp(self.logvar / 2.) * eps
        else:
            if self.train_gmm:
                if hold_seed is not None:
                    torch.random.manual_seed(hold_seed)
                    logits = self.logits.reshape([1, 1, self.n_mixtures]).repeat(1, self.max_outputs, 1)  # [1, N, M]
                    onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True).repeat(bsize, 1, 1).unsqueeze(-1)  # [B, N, M, 1]
                else:
                    logits = self.logits.reshape([1, 1, self.n_mixtures]).repeat(bsize, self.max_outputs, 1)  # [B, N, M]
                    onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True).unsqueeze(-1)  # [B, N, M, 1]
                mu = self.mu.reshape([1, 1, self.n_mixtures, self.dim_seed])  # [1, 1, M, D]
                sig = self.sig.reshape([1, 1, self.n_mixtures, self.dim_seed])  # [1, 1, M, D]
                mu = (mu * onehot).sum(2)  # [B, N, D]
                sig = (sig * onehot).sum(2)  # [B, N, D]
                x = mu + sig * eps
            else:
                mix = D.Categorical(self.logits)
                comp = D.Independent(D.Normal(self.mu, self.sig.abs()), 1)
                mixture = D.MixtureSameFamily(mix, comp)
                x = mixture.sample((output_sizes.size(0), self.max_outputs))

        x = self.output(x)  # [B, N, D]
        return x, x_mask

    def multi_gpu_wrapper(self, f):
        if self.n_mixtures == 1:
            self.mu = f(self.mu)
            self.logvar = f(self.logvar)
        else:
            self.logits = f(self.logits)
            self.mu = f(self.mu)
            self.sig = f(self.sig)
            self.mixture = f(self.mixture)
        self.output = f(self.output)


class ResidualAttention(nn.Module):
    """An adaptation of MAB in Set Transformer"""
    def __init__(self, dim_q, dim_k, dim_v, num_heads, ln=False, dropout_p=0., slot_att=False):
        super().__init__()
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.dim_split = dim_v // num_heads if dim_v >= num_heads else 1
        self.dropout_p = dropout_p
        self.slot_att = slot_att
        self.fc_q = nn.Linear(dim_q, dim_v, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_v, bias=False)
        self.fc_v = nn.Linear(dim_k, dim_v, bias=False)
        self.fc_o = nn.Linear(dim_v, dim_q, bias=False)
        self.ffn1 = nn.Linear(dim_q, dim_q)
        self.ffn2 = nn.Linear(dim_q, dim_q)
        if ln:
            self.ln_x = nn.LayerNorm(dim_q)
            self.ln_y = nn.LayerNorm(dim_k)
            self.ln_o1 = nn.LayerNorm(dim_q)
            self.ln_o2 = nn.LayerNorm(dim_q)
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

    def compute_attention(self, query, key, value, x_mask, y_mask, given_alpha):
        bs, xs, ys = query.shape[0], query.shape[1], value.shape[1]
        if (x_mask is not None) and (y_mask is not None):
            mask = torch.logical_or(x_mask.reshape([1, bs, xs, 1]).repeat(1, 1, 1, ys),
                                    y_mask.reshape([1, bs, 1, ys]).repeat(1, 1, xs, 1))
        elif x_mask is None:
            mask = None if (y_mask is None) else y_mask.reshape([1, bs, 1, ys]).repeat(1, 1, xs, 1)
        else:  # (y_mask is None) and (x_mask is not None)
            mask = x_mask.reshape([1, bs, xs, 1]).repeat(1, 1, 1, ys)
        q_ = torch.cat(query.split(self.dim_split, 2), 0)  # [H * B, N, Dh]
        k_ = torch.cat(key.split(self.dim_split, 2), 0)  # [H * B, M, Dh]
        v_ = torch.cat(value.split(self.dim_split, 2), 0)  # [H * B, M, Dh]
        sdp = q_.bmm(k_.transpose(1, 2)) / math.sqrt(self.dim_v)  # [H * B, N, M]
        sdp_ = torch.stack(sdp.split(bs, 0), 0).clone()  # [H, B, N, M]
        sdp_ = sdp_.masked_fill(mask, -1e38) if (mask is not None) else sdp_
        sdp = sdp_.flatten(0, 1)  # [H * B, N, M]
        alpha = torch.softmax(sdp, 1 if self.slot_att else 2).clone()  # [H * B, N, M]
        alpha = given_alpha.flatten(0, 1) if (given_alpha is not None) else alpha  # [H * B, N, M]
        att = alpha.bmm(v_)  # [H * B, N, Dh]
        if self.slot_att:  # weighted mean
            att /= (alpha + 1e-5).sum(dim=-1).unsqueeze(-1)
        att = torch.cat(att.split(bs, 0), 2)  # [B, N, Dv]
        att = masked_fill(att, x_mask, 0.)
        return att, alpha

    def forward(self, x, y, x_mask, y_mask, get_alpha=False, given_alpha=None):
        """ MAB(x, y) = LayerNorm(h + rFF(h)), where h = LayerNorm(x + MultiHead(query=x, key=y, value=y)).
        Ignore elements specified by masks
        :param x: Tensor([B, N, C])
        :param y: Tensor([B, M, D])
        :param x_mask: BoolTensor([B, N])
        :param y_mask: BoolTensor([B, M])
        :param get_alpha
        :param given_alpha: Tensor([H, B, N, M])
        :return: Tensor([B, N, C]), possibly with Tensor([H, B, N, M])
        """
        x = masked_fill(x.clone(), x_mask, 0.)
        y = masked_fill(y.clone(), y_mask, 0.)
        q = self.fc_q(x)  # [B, N, Dv]
        k = self.fc_k(y)  # [B, M, Dv]
        v = self.fc_v(y)  # [B, M, Dv]
        q = masked_fill(q, x_mask, 0.)
        k = masked_fill(k, y_mask, 0.)
        v = masked_fill(v, y_mask, 0.)
        att, alpha = self.compute_attention(q, k, v, x_mask, y_mask, given_alpha)  # [B, N, Dv]
        att = self.fc_o(att)  # post,2 in PreLN Table 1
        att = self.dropout(att) if hasattr(self, 'dropout') else att
        o = x + att
        o = self.ln_o1(o) if hasattr(self, 'ln_o1') else o  # post, 3
        ff = self.ffn2(F.relu(self.ffn1(o)))  # post, 5 = post, 3 + post, 4
        ff = self.dropout(ff) if hasattr(self, 'dropout') else ff
        o = o + ff
        o = self.ln_o2(o) if hasattr(self, 'ln_o2') else o
        o = masked_fill(o, x_mask, 0.)
        if get_alpha:
            alpha = torch.stack(alpha.split(x.shape[0], 0), 0)  # [H, B, N, M]
            return o, alpha
        return o


class AttentiveBlock(nn.Module):
    """Base class of ISAB in Set Transformer and ABL (Attentive Bottleneck Layer)"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout_p=0., slot_att=False):
        super().__init__()
        self.num_inds = num_inds
        self.register_parameter(name='i', param=nn.Parameter(torch.randn(1, num_inds, dim_out)))  # [1, M, D]
        nn.init.xavier_uniform_(self.i)
        self.att1 = ResidualAttention(dim_out, dim_in, dim_out, num_heads, ln, dropout_p, slot_att)
        self.att2 = ResidualAttention(dim_in, dim_out, dim_out, num_heads, ln, dropout_p)

    def project(self, x, x_mask=None):
        i = self.i.repeat(x.shape[0], 1, 1)  # [B, M, D]
        h, alpha = self.att1(i, x, None, x_mask, get_alpha=True)  # [B, M, D], [H, B, M, N]
        return h, alpha.transpose(2, 3)  # [B, M, D], [H, B, N, M]

    def broadcast(self, h, x, x_mask=None):
        o, alpha = self.att2(x, h, x_mask, None, get_alpha=True)  # [B, N, D], [H, B, N, M]
        return o, alpha

    def forward(self, x, x_mask=None):
        """ Deterministic forward pass
        :param x: Tensor([B, N, Di])
        :param x_mask: BoolTensor([B, N])
        :return: Tensor([B, N, D]), Tensor([B, K, D]), Tuple(Tensor([H, B, N, M]), Tensor([H, B, N, M]))
        """
        h, alpha1 = self.project(x, x_mask)
        o, alpha2 = self.broadcast(h, x, x_mask)
        return o, h, alpha1, alpha2


class ISAB(AttentiveBlock):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln, dropout_p=0.):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln=ln, dropout_p=dropout_p, slot_att=False)

    def forward(self, x, x_mask=None):
        h, alpha1 = self.project(x, x_mask)
        o, alpha2 = self.broadcast(h, x, x_mask)
        return o
