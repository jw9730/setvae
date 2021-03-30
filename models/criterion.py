import torch
import torch.nn as nn

from metrics import chamfer_loss, emd_loss


class ChamferCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()

    @staticmethod
    def forward(output, target_set, target_mask, args, epoch=None):
        """
        Compute loss
        :param output
        :param target_set: Tensor([B, N, 2])
        :param target_mask: Tensor([B, N])
        :param args
        :param epoch
        """
        output_set, output_mask, kls = output['set'], output['set_mask'], output['kls']
        assert (~output_mask).sum() == (~target_mask).sum(), f"{(~output_mask).sum(-1)} != {(~target_mask).sum(-1)}"
        accelerate = (output_set.shape[-1] == 3) and args.device == 'cuda'
        l2_loss = chamfer_loss(output_set, output_mask, target_set, target_mask, accelerate=accelerate)  # [B,] -> scalar
        kl_loss = torch.stack(kls, dim=1).sum(dim=1).mean()  # [B, Dz] -> scalar
        if args.kl_warmup_epochs > 0:
            assert epoch is not None
            beta = args.beta * min(1, epoch / args.kl_warmup_epochs)
        else:
            beta = args.beta
        loss = beta * kl_loss + l2_loss
        topdown_kl = [kl.detach().mean(dim=0) / float(scale * args.z_dim) for scale, kl in zip(args.z_scales, kls)]
        return {'loss': loss,
                'kl': kl_loss, 'l2': l2_loss,
                'topdown_kl': topdown_kl,
                'beta': beta}


class EMDCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()

    @staticmethod
    def forward(output, target_set, target_mask, args, epoch=None):
        """
        Compute loss
        :param output
        :param target_set: Tensor([B, N, 2])
        :param target_mask: Tensor([B, N])
        :param args
        :param epoch
        """
        output_set, output_mask, kls = output['set'], output['set_mask'], output['kls']
        assert (~output_mask).sum() == (~target_mask).sum(), f"{(~output_mask).sum(-1)} != {(~target_mask).sum(-1)}"
        assert (output_set.shape[-1] == 3) and args.device == 'cuda'
        l1_loss = emd_loss(output_set, output_mask, target_set, target_mask)  # [B,] -> scalar
        kl_loss = torch.stack(kls, dim=1).sum(dim=1).mean()  # [B, Dz] -> scalar
        if args.kl_warmup_epochs > 0:
            assert epoch is not None
            beta = args.beta * min(1, epoch / args.kl_warmup_epochs)
        else:
            beta = args.beta
        loss = beta * kl_loss + l1_loss
        topdown_kl = [kl.detach().mean(dim=0) / float(scale * args.z_dim) for scale, kl in zip(args.z_scales, kls)]
        return {'loss': loss,
                'kl': kl_loss, 'l2': l1_loss,  # abuse of loss name for handling convenience
                'topdown_kl': topdown_kl,
                'beta': beta}


class CombinedCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()

    @staticmethod
    def forward(output, target_set, target_mask, args, epoch=None):
        """
        Compute loss
        :param output
        :param target_set: Tensor([B, N, 2])
        :param target_mask: Tensor([B, N])
        :param args
        :param epoch
        """
        output_set, output_mask, kls = output['set'], output['set_mask'], output['kls']
        assert (~output_mask).sum() == (~target_mask).sum(), f"{(~output_mask).sum(-1)} != {(~target_mask).sum(-1)}"
        accelerate = (output_set.shape[-1] == 3) and args.device == 'cuda'
        cd = chamfer_loss(output_set, output_mask, target_set, target_mask, accelerate=accelerate)  # [B,] -> scalar
        assert (output_set.shape[-1] == 3) and args.device == 'cuda'
        emd = emd_loss(output_set, output_mask, target_set, target_mask)  # [B,] -> scalar
        recon_loss = cd + emd
        kl_loss = torch.stack(kls, dim=1).sum(dim=1).mean()  # [B, Dz] -> scalar
        if args.kl_warmup_epochs > 0:
            assert epoch is not None
            beta = args.beta * min(1, epoch / args.kl_warmup_epochs)
        else:
            beta = args.beta
        loss = beta * kl_loss + recon_loss
        topdown_kl = [kl.detach().mean(dim=0) / float(scale * args.z_dim) for scale, kl in zip(args.z_scales, kls)]
        return {'loss': loss,
                'kl': kl_loss, 'l2': recon_loss,  # abuse of loss name for handling convenience
                'topdown_kl': topdown_kl,
                'beta': beta}
