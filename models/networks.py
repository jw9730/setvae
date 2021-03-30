import torch
import torch.nn as nn
import torch.optim as optim

from .layers import AttentiveBlock, InitialSet, ResidualAttention, ISAB
from .mlp import MLP, ElementwiseMLP
from .criterion import ChamferCriterion, EMDCriterion, CombinedCriterion
from .ops import get_module


class DeterministicNetwork(nn.Module):
    def __init__(self, net_type, num_inds, dim, dim_out, n_hidden, num_heads=0, ln=False, dropout_p=0.,
                 activation='relu', use_bn=False, residual=False):
        super().__init__()
        self.num_inds = num_inds
        self.net_type = net_type
        if self.net_type == 'elem_mlp':
            self.net = ElementwiseMLP(dim, dim, dim_out, n_hidden, activation, residual, use_bn, use_masked_bn=True, dropout_p=dropout_p)
        elif self.net_type == 'set_transformer':
            self.net = nn.ModuleList()
            for i in range(n_hidden):
                self.net.append(ISAB(dim, dim_out, num_heads, num_inds, ln=ln, dropout_p=dropout_p))

    def forward(self, x, x_mask):
        if self.net_type == 'elem_mlp':
            return self.net(x, x_mask)
        elif self.net_type == 'set_transformer':
            for layer in self.net:
                x = layer(x, x_mask)
            return x


class InducedNetwork(nn.Module):
    def __init__(self, net_type, num_inds, dim_in, dim_hidden, dim_out, n_hidden, num_heads=0, ln=False, dropout_p=0.):
        super().__init__()
        self.num_inds = num_inds
        self.net_type = net_type
        if self.net_type == 'full_mlp':
            self.net = MLP(num_inds*dim_in, dim_hidden, num_inds*dim_out, n_hidden, dropout_p=dropout_p)
        elif self.net_type == 'elem_mlp':
            self.net = ElementwiseMLP(dim_in, dim_hidden, dim_out, n_hidden, dropout_p=dropout_p)
        elif self.net_type == 'set_transformer':
            self.input = nn.Linear(dim_in, dim_hidden)
            self.net = nn.ModuleList()
            for i in range(n_hidden):
                self.net.append(ResidualAttention(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln, dropout_p=dropout_p))
            self.output = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        bs, num_inds, dim_in = x.shape
        assert num_inds == self.num_inds
        if self.net_type == 'full_mlp':
            return self.net(x.reshape([bs, -1])).reshape([bs, num_inds, -1])
        elif self.net_type == 'elem_mlp':
            return self.net(x)
        elif self.net_type == 'set_transformer':
            x = self.input(x)
            for layer in self.net:
                x = layer(x, x, None, None)
            return self.output(x)


class EncoderBlock(AttentiveBlock):
    """ISAB in Set Transformer"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln, dropout_p, slot_att):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln, dropout_p, slot_att)


class DecoderBlock(AttentiveBlock):
    """ABL (Attentive Bottleneck Layer)"""
    def __init__(self, dim_in, dim_out, dim_z, num_heads, num_inds, ln, dropout_p,
                 slot_att, i_net, i_net_layers, cond_prior=True):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln, dropout_p, slot_att)
        self.cond_prior = cond_prior
        if cond_prior:
            self.prior = InducedNetwork(i_net, num_inds, dim_out, dim_out, 2*dim_z, i_net_layers, num_heads, ln, dropout_p)
        else:
            self.register_parameter(name='prior', param=nn.Parameter(torch.randn(1, num_inds, 2*dim_z)))  # [1, M, 2Dz]
            nn.init.xavier_uniform_(self.prior)
        self.posterior = InducedNetwork(i_net, num_inds, dim_out, dim_out, 2*dim_z, i_net_layers, num_heads, ln, dropout_p)
        self.fc = nn.Linear(dim_z, dim_out)

    def compute_prior(self, h):
        """
        Sample from prior
        :param h: Tensor([B, M, D])
        :return: Tensor([B, M, Dz])
        """
        bs, num_inds, dim_in = h.shape
        if self.cond_prior:  # [B, M, 2Dz]
            prior = self.prior(h)
        else:
            prior = self.prior.repeat(bs, 1, 1)
        mu = prior[..., :prior.shape[-1]//2]  # [B, M, Dz]
        logvar = prior[..., prior.shape[-1]//2:].clamp(-4., 3.)
        eps = torch.randn(mu.shape).to(h)
        z = mu + torch.exp(logvar / 2.) * eps  # [B, M, Dz]
        return z, mu, logvar

    def compute_posterior(self, mu, logvar, bottom_up_h, h=None):
        """
        Estimate residual posterior parameters from prior parameters and top-down features
        :param mu: Tensor([B, M, D])
        :param logvar: Tensor([B, M, D])
        :param bottom_up_h: Tensor([B, M, D])
        :param h: Tensor([B, M, D])
        :return: Tensor([B, M, Dz]), Tensor([B, M, Dz])
        """
        bs, num_inds, dim_in = bottom_up_h.shape
        assert self.num_inds == num_inds
        bottom_up_h = bottom_up_h + h if h is not None else bottom_up_h
        posterior = self.posterior(bottom_up_h)
        mu2 = posterior[..., :posterior.shape[-1]//2]  # [B, M, Dz]
        logvar2 = posterior[..., posterior.shape[-1]//2:].clamp(-4., 3.)
        sigma = torch.exp(logvar / 2.)
        sigma2 = torch.exp(logvar2 / 2.)
        eps = torch.randn(mu.shape).to(mu)
        z = (mu + mu2) + (sigma * sigma2) * eps
        kl = -0.5 * (logvar2 + 1. - mu2.pow(2) / sigma.pow(2) - sigma2.pow(2)).view(mu.shape[0], -1).sum(dim=-1)  # [B,]
        return z, kl, mu2, logvar2

    def broadcast_latent(self, z, h, x, x_mask=None):
        return self.broadcast(self.fc(z), x, x_mask)  # No residual


class SetVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpu = args.gpu
        self.input_dim = args.input_dim
        self.max_outputs = args.max_outputs
        self.fixed_gmm = args.fixed_gmm
        self.train_gmm = args.train_gmm
        self.init_dim = args.init_dim
        self.n_mixtures = args.n_mixtures
        self.n_layers = len(args.z_scales)
        self.z_dim = args.z_dim
        self.z_scales = args.z_scales
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.slot_att = args.slot_att
        self.i_net = args.i_net
        self.i_net_layers = args.i_net_layers
        self.d_net = args.d_net
        self.enc_in_layers = args.enc_in_layers
        self.dec_in_layers = args.dec_in_layers
        self.dec_out_layers = args.dec_out_layers
        self.isab_inds = args.isab_inds
        self.ln = args.ln
        self.dropout_p = args.dropout_p
        self.activation = args.activation
        self.use_bn = args.use_bn
        self.residual = args.residual
        self.enc_inds = list(reversed(self.z_scales))  # bottom-up
        self.dec_inds = self.z_scales  # top-down
        self.input = nn.Linear(self.input_dim, self.hidden_dim)
        self.init_set = InitialSet(args, self.init_dim, self.n_mixtures, self.fixed_gmm, self.train_gmm, self.hidden_dim, self.max_outputs)
        self.pre_encoder = DeterministicNetwork(self.d_net, self.isab_inds, self.hidden_dim, self.hidden_dim,
                                                self.enc_in_layers, self.num_heads, self.ln, self.dropout_p,
                                                self.activation, self.use_bn, self.residual)
        self.pre_decoder = DeterministicNetwork(self.d_net, self.isab_inds, self.hidden_dim, self.hidden_dim,
                                                self.dec_in_layers, self.num_heads, self.ln, self.dropout_p,
                                                self.activation, self.use_bn, self.residual)
        self.post_decoder = DeterministicNetwork(self.d_net, self.isab_inds, self.hidden_dim, self.hidden_dim,
                                                 self.dec_out_layers, self.num_heads, self.ln, self.dropout_p,
                                                 self.activation, self.use_bn, self.residual)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(EncoderBlock(self.hidden_dim, self.hidden_dim, self.num_heads,
                                             self.enc_inds[i], self.ln, self.dropout_p, self.slot_att))
            self.decoder.append(DecoderBlock(self.hidden_dim, self.hidden_dim, self.z_dim, self.num_heads,
                                             self.dec_inds[i], self.ln, self.dropout_p, self.slot_att, self.i_net,
                                             self.i_net_layers, cond_prior=i > 0))
        self.output = nn.Linear(self.hidden_dim, self.input_dim)

    def bottom_up(self, x, x_mask):
        """ Deterministic bottom-up encoding
        :param x: Tensor([B, N, Di])
        :param x_mask: BoolTensor([B, N])
        :return: List([Tensor([B, M, D])]), List([Tensor([H, B, N, M]), Tensor([H, B, N, M])])
        """
        x = self.input(x)  # [B, N, D]
        x = self.pre_encoder(x, x_mask)
        features = list()
        alphas = list()
        for layer in get_module(self.gpu, self.encoder):
            x, h, alpha1, alpha2 = layer(x, x_mask)  # [B, N, D], [B, M, D], [H, B, N, M], [H, B, N, M]
            features.append(h)
            alphas.append((alpha1, alpha2))
        return {'features': features, 'alphas': alphas}

    def top_down(self, cardinality, bottom_up_h):
        """ Stochastic top-down decoding
        :param cardinality: Tensor([B,])
        :param bottom_up_h: List([Tensor([B, M, D])]) in top-down order
        :return:
        """
        o, o_mask = self.init_set(cardinality)
        o = self.pre_decoder(o, o_mask)
        alphas, posteriors, kls = [], [(o, None, None)], []
        for idx, layer in enumerate(get_module(self.gpu, self.decoder)):
            h, alpha1 = layer.project(o, o_mask)
            _, mu, logvar = layer.compute_prior(h)
            z, kl, mu2, logvar2 = layer.compute_posterior(mu, logvar, bottom_up_h[idx], None if idx == 0 else h)
            o, alpha2 = layer.broadcast_latent(z, h, o, o_mask)
            alphas.append((alpha1, alpha2))
            posteriors.append((z, mu2, logvar2))
            kls.append(kl)
        o = self.post_decoder(o, o_mask)
        o = self.output(o)  # [B, N, Do]
        return {'set': o, 'set_mask': o_mask,
                'posteriors': posteriors, 'kls': kls, 'alphas': alphas}

    def forward(self, x, x_mask):
        """ Bidirectional inference
        :param x: Tensor([B, N, Di])
        :param x_mask: BoolTensor([B, N])
        :return: Tensor([B, N, Do]), Tensor([B, N]), List([Tensor([H, B, N, M]), Tensor([H, B, N, M])]) * 2
        """
        bup = self.bottom_up(x, x_mask)
        tdn = self.top_down((~x_mask).sum(-1), list(reversed(bup['features'])))
        o, o_mask = self.postprocess(tdn['set'], tdn['set_mask'])
        return {'set': o, 'set_mask': o_mask,
                'posteriors': tdn['posteriors'], 'kls': tdn['kls'],
                'alphas': (bup['alphas'], tdn['alphas'])}

    def sample(self, output_sizes, hold_seed=None, hold_initial_set=False, given_latents=None):
        """ Top-down generation
        :param output_sizes: Tensor([B,])
        :param hold_seed
        :param hold_initial_set
        :param given_latents: List([Tensor([B, ?, D])])
        :return: Tensor([B, N, Do]), Tensor([B, N]), List([Tensor([B, M, D])]),
                 List([Tensor([H, B, N, M]), Tensor([H, B, N, M])])
        """
        o, o_mask = self.init_set(output_sizes, hold_seed, hold_initial_set)
        o = self.pre_decoder(o, o_mask)
        priors = [(o, None, None)]
        # if given_latents is not None:
        #    o = given_latents[0]
        #    assert o.shape[1] == self.max_outputs
        alphas = list()
        for idx, layer in enumerate(get_module(self.gpu, self.decoder)):
            h, alpha1 = layer.project(o, o_mask)
            if idx == 0:
                z, mu, logvar = layer.compute_prior(h)
            z, mu, logvar = layer.compute_prior(h)
            if given_latents is not None:
                z = given_latents[idx + 1]
                assert z.shape[1] == mu.shape[1]
            o, alpha2 = layer.broadcast_latent(z, h, o, o_mask)
            priors.append((z, mu, logvar))
            alphas.append((alpha1, alpha2))
        o = self.post_decoder(o, o_mask)
        o = self.output(o)  # [B, N, Do]
        o, o_mask = self.postprocess(o, o_mask)
        return {'set': o, 'set_mask': o_mask,
                'priors': priors, 'alphas': alphas}

    @staticmethod
    def postprocess(x, x_mask):
        if x.shape[-1] == 2:  # MNIST, xy
            return (torch.tanh(x) + 1) / 2., x_mask  # [B, N, Do], [0, 1] range
        elif x.shape[-1] == 3:  # ShapeNet, xyz
            return x, x_mask  # [B, N, Do]
        elif x.shape[-1] == 4:  # KITTI, xyzc
            x = x.clone()
            x[..., -1] = (torch.tanh(x[..., -1]) + 1) / 2.
            return x, x_mask  # [B, N, Do]

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(self.parameters())

        if args.matcher == 'hungarian':
            criterion = EMDCriterion(args)
        elif args.matcher == 'chamfer':
            criterion = ChamferCriterion(args)
        else:
            assert args.matcher == 'all'
            criterion = CombinedCriterion(args)

        return opt, criterion

    def multi_gpu_wrapper(self, f):
        self.input = f(self.input)
        self.encoder = f(self.encoder)
        self.init_set = f(self.init_set)
        self.decoder = f(self.decoder)
        self.output = f(self.output)
