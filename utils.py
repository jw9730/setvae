import os
import random
import numpy as np
from pathlib import Path
from pprint import pprint
import time

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import torch


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pad(x, x_mask, max_size):
    if x.size(1) < max_size:
        pad_size = max_size - x.size(1)
        pad = torch.ones(x.size(0), pad_size, x.size(2)).to(x.device) * float('inf')
        pad_mask = torch.ones(x.size(0), pad_size).bool().to(x.device)
        x, x_mask = torch.cat((x, pad), dim=1), torch.cat((x_mask, pad_mask), dim=1)
    else:
        UserWarning(f"pad: {x.size(1)} >= {max_size}")
    return x, x_mask


def extend_batch(b, b_mask, x, x_mask):
    if b is None:
        return x, x_mask
    if b.size(1) >= x.size(1):
        x, x_mask = pad(x, x_mask, b.size(1))
    else:
        b, b_mask = pad(b, b_mask, x.size(1))
    return torch.cat((b, x), dim=0), torch.cat((b_mask, x_mask), dim=0)


@torch.no_grad()
def draw_mnist(x: torch.Tensor, x_mask: torch.Tensor):
    """ Make MNIST image
    :param x: Tensor([B, N, 2])
    :param x_mask: Tensor([B, N])
    :return: Tensor([3 * B, H, W])
    """
    tic = time.time()
    figw, figh = 16., 16.
    W, H = 256, int(256 * figh / figw)

    imgs = list()
    for p, m in zip(x, x_mask):
        p = p[~m, :]
        p = p.cpu()

        fig = plt.figure(figsize=(figw, figh))
        ax = fig.gca()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor((0., 0., 0.))

        ax.scatter(p[:, 0], -p[:, 1], color=(1, 1, 1), s=500, ec=(.2, .2, .2))

        plt.xlim(0, 1)
        plt.ylim(-1, 0)
        fig.tight_layout()
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        img = np.frombuffer(buf, np.uint8).copy()
        img.shape = int(h), int(w), 4
        img = img[:, :, 0:3]
        img = cv2.resize(img, dsize=(H, int(w * (float(H) / h))), interpolation=cv2.INTER_CUBIC)  # [H, W, 3]
        imgs.append(torch.tensor(img).transpose(2, 0))  # [3, H, W]

        fig.canvas.flush_events()
        plt.close(fig)

    return torch.stack(imgs, dim=0)


@torch.no_grad()
def draw_attention_mnist(x, x_mask, alpha):
    """ Compute batched images of attention weight for each attention head, with different color for each inducing point
    :param x: Tensor([B, N, 2])
    :param x_mask: Tensor([B, N])
    :param alpha: Tensor([n_heads, B, N, I])
    :return: List(Tensor([3, H, n_heads * W]))
    """
    tic = time.time()
    n_heads, bsize, N, isize = alpha.shape
    H, W = 256, 256

    palette = plt.get_cmap('hsv')(np.linspace(0., 1., isize, endpoint=False))[:, :-1]

    imgs = list()
    for a, p, m in zip(alpha.unbind(1), x, x_mask):  # [n_heads, N, I], [N, 2], [N,]
        a = a[:, ~m, :]  # [n_heads, M, I]
        p = p[~m, :]  # [M, 2]
        a = a.cpu()
        p = p.cpu()
        for a_h in a.unbind(0):  # [M, I]
            fig = plt.figure(figsize=(16, 16))
            ax = fig.gca()
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor((0., 0., 0.))

            rgb = np.asarray([[palette[i, :]] for i in range(isize)])  # [I, 1, 3]
            hsv = rgb_to_hsv(rgb)
            alphas = a_h.transpose(1, 0).unsqueeze(2)  # [I, M, 1]
            alphas = np.asarray(alphas / (alphas.sum(dim=0) + 1e-5))
            alphas[alphas < 0] = 0.
            alphas[alphas > 1] = 1.
            hsv = (hsv * alphas).sum(axis=0)  # [M, 3]
            rgb = hsv_to_rgb(hsv)
            rgb[rgb < 0] = 0.
            rgb[rgb > 1] = 1.
            ax.scatter(p[:, 0], -p[:, 1], c=rgb, s=500, ec=None)

            plt.xlim(0, 1)
            plt.ylim(-1, 0)
            fig.tight_layout()
            fig.canvas.draw()

            buf = fig.canvas.buffer_rgba()
            l, b, w, h = fig.bbox.bounds
            img = np.frombuffer(buf, np.uint8).copy()
            img.shape = int(h), int(w), 4
            img = img[:, :, 0:3]
            img = cv2.resize(img, dsize=(H, int(w * (float(H) / h))), interpolation=cv2.INTER_CUBIC)  # [H, W, 3]
            img = torch.tensor(img).transpose(2, 0)  # [3, H, W]
            imgs.append(img)

            fig.canvas.flush_events()
            plt.close(fig)

    imgs = torch.stack(imgs, dim=0).reshape([bsize, n_heads, 3, H, W])  # [B, n_heads, 3, H, W]

    composed = list()
    for img_h in imgs:  # [n_heads, 3, H, W]
        img = torch.cat(img_h.unbind(0), dim=-1)  # [3, H, n_heads * W]
        composed.append(img)

    return composed  # bsize * Tensor([3, H, n_heads * W])


@torch.no_grad()
def draw_pointcloud(x: torch.Tensor, x_mask: torch.Tensor, grid_on=True):
    """ Make point cloud image
    :param x: Tensor([B, N, 3])
    :param x_mask: Tensor([B, N])
    :param grid_on
    :return: Tensor([3 * B, W, H])
    """
    tic = time.time()
    figw, figh = 16., 12.
    W, H = 256, int(256 * figh / figw)

    imgs = list()
    for p, m in zip(x, x_mask):
        p = p[~m, :]
        p = p.cpu()

        fig = plt.figure(figsize=(figw, figh))
        ax = fig.gca(projection='3d')
        ax.set_facecolor('xkcd:steel')
        ax.w_xaxis.set_pane_color((0., 0., 0., 1.0))
        ax.w_yaxis.set_pane_color((0., 0., 0., 1.0))
        ax.w_zaxis.set_pane_color((0., 0., 0., 1.0))

        ax.scatter(-p[:, 2], p[:, 0], p[:, 1], color=(1, 1, 1), marker='o', s=100)
        fig.tight_layout()
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        img = np.frombuffer(buf, np.uint8).copy()
        img.shape = int(h), int(w), 4
        img = img[:, :, 0:3]
        img = cv2.resize(img, dsize=(W, H), interpolation=cv2.INTER_CUBIC)  # [H, W, 3]

        imgs.append(torch.tensor(img).transpose(2, 0).transpose(2, 1))  # [3, H, W]
        plt.close(fig)

    return torch.stack(imgs, dim=0)


@torch.no_grad()
def draw_attention_pointcloud(x, x_mask, alpha):
    """ Compute batched images of attention weight, grid over inducing points and attention heads
    :param x: Tensor([B, N, 3])
    :param x_mask: Tensor([B, N])
    :param alpha: Tensor([n_heads, B, N, I])
    :return: List(Tensor([3, n_heads * W, I * H]))
    """
    tic = time.time()
    figw, figh = 16., 12.
    W, H = 256, int(256 * figh / figw)
    n_heads, bsize, N, isize = alpha.shape

    palette = plt.get_cmap('hsv')(np.linspace(0., 1., isize, endpoint=False))[:, :-1]

    imgs = list()
    for a, p, m in zip(alpha.unbind(1), x, x_mask):  # [n_heads, N, I], [N, 2], [N,]
        a = a[:, ~m, :]  # [n_heads, M, I]
        p = p[~m, :]  # [M, 2]
        a = a.cpu()
        p = p.cpu()
        for a_h in a.unbind(0):  # [M, I]
            fig = plt.figure(figsize=(figw, figh))
            ax = fig.gca(projection='3d')
            ax.set_facecolor('xkcd:steel')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.w_xaxis.set_pane_color((0., 0., 0., 1.0))
            ax.w_yaxis.set_pane_color((0., 0., 0., 1.0))
            ax.w_zaxis.set_pane_color((0., 0., 0., 1.0))

            rgb = np.asarray([[palette[i, :]] for i in range(isize)])  # [I, 1, 3]
            hsv = rgb_to_hsv(rgb)
            alphas = a_h.transpose(1, 0).unsqueeze(2)  # [I, M, 1]
            alphas = np.asarray(alphas / (alphas.sum(dim=0) + 1e-5))
            alphas[alphas < 0] = 0.
            alphas[alphas > 1] = 1.
            hsv = (hsv * alphas).sum(axis=0)  # [M, 3]
            rgb = hsv_to_rgb(hsv)
            rgb[rgb < 0] = 0.
            rgb[rgb > 1] = 1.

            ax.scatter(-p[:, 2], p[:, 0], p[:, 1], c=rgb, marker='o', s=100)
            fig.tight_layout()
            fig.canvas.draw()

            buf = fig.canvas.buffer_rgba()
            l, b, w, h = fig.bbox.bounds
            img = np.frombuffer(buf, np.uint8).copy()
            img.shape = int(h), int(w), 4
            img = img[:, :, 0:3]
            img = cv2.resize(img, dsize=(W, H), interpolation=cv2.INTER_CUBIC)  # [H, W, 3]

            imgs.append(torch.tensor(img).transpose(2, 0).transpose(2, 1))  # [3, H, W]
            plt.close(fig)

    imgs = torch.stack(imgs, dim=0).reshape([bsize, n_heads, 3, H, W])  # [B, n_heads, 3, H, W]

    composed = list()
    for img_h in imgs:  # [n_heads, 3, H, W]
        img = torch.cat(img_h.unbind(0), dim=-1)  # [3, H, n_heads * W]
        composed.append(img)

    return composed  # bsize * Tensor([3, H, n_heads * W])


@torch.no_grad()
def draw(x, x_mask):
    if x.size(-1) == 2:
        return draw_mnist(x, x_mask)
    elif x.size(-1) == 3:
        return draw_pointcloud(x, x_mask)
    else:
        raise NotImplementedError


@torch.no_grad()
def draw_attention(x, x_mask, alpha):
    if x.size(-1) == 2:
        return draw_attention_mnist(x, x_mask, alpha)
    elif x.size(-1) == 3:
        return draw_attention_pointcloud(x, x_mask, alpha)
    else:
        raise NotImplementedError


@torch.no_grad()
def validate_reconstruct_l2(epoch, val_loader, model, criterion, args, logger):
    start_time = time.time()
    model.train()
    criterion.train()
    l2_meter = AverageValueMeter()
    kl_meter = AverageValueMeter()

    for bidx, data in enumerate(val_loader):
        bsize = data['set_mask'].size(0)
        idx_b, gt, gt_mask = data['idx'], data['set'], data['set_mask']
        gt = gt.cuda() if args.gpu is None else gt.cuda(args.gpu)
        gt_mask = gt_mask.to(gt.device)

        output = model(gt, gt_mask)
        recon, recon_mask = output['set'], output['set_mask']

        if args.denormalized_loss:
            # denormalize
            try:
                m, s = data['mean'].float(), data['std'].float()
                m = m.to(gt.device)
                s = s.to(gt.device)
            except (TypeError, AttributeError) as e:
                m, s = float(data['mean']), float(data['std'])

            if args.standardize_per_shape:
                offset = data['offset']
                gt = gt + offset.to(gt.device)
                recon = recon + offset.to(recon.device)

            recon = recon * s + m
            gt = gt * s + m

        losses = criterion(output, gt, gt_mask, args, epoch)
        loss, kl_loss, recon_loss, topdown_kl, beta = losses['loss'], losses['kl'], losses['l2'], losses['topdown_kl'], \
                                                      losses['beta']

        l2_meter.update(recon_loss.detach().item(), bsize)
        kl_meter.update(kl_loss.detach().item(), bsize)

        if bidx % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()
            print("[Rank %d] <VAL> Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss %2.5f KL %2.5f Recon %2.5f"
                  % (args.local_rank, epoch, bidx, len(val_loader), duration,
                     loss.detach().item(), kl_loss.detach().item(), recon_loss.detach().item()))

    if logger is not None:
        logger.add_scalar('VAL kl loss (epoch)', kl_meter.avg, epoch)
        logger.add_scalar('VAL recon loss (epoch)', l2_meter.avg, epoch)
        print('Log sent')
    return kl_meter.avg, l2_meter.avg


@torch.no_grad()
def validate_reconstruct(loader, model, args, max_samples, save_dir):
    from metrics import emd_cd_masked
    all_sample, all_sample_mask = None, None
    all_ref, all_ref_mask = None, None
    all_idx = list()
    ttl_samples = 0

    iterator = iter(loader)
    for data in iterator:
        idx_b, gt, gt_mask = data['idx'], data['set'], data['set_mask']
        gt = gt.cuda() if args.gpu is None else gt.cuda(args.gpu)
        gt_mask = gt_mask.to(gt.device)

        output = model(gt, gt_mask)
        recon, recon_mask = output['set'], output['set_mask']

        # denormalize
        try:
            m, s = data['mean'].float(), data['std'].float()
            m = m.to(gt.device)
            s = s.to(gt.device)
        except (TypeError, AttributeError) as e:
            m, s = float(data['mean']), float(data['std'])

        if args.standardize_per_shape:
            offset = data['offset']
            gt = gt + offset.to(gt.device)
            recon = recon + offset.to(recon.device)

        recon = recon * s + m
        gt = gt * s + m

        all_sample, all_sample_mask = extend_batch(all_sample, all_sample_mask, recon, recon_mask)
        all_ref, all_ref_mask = extend_batch(all_ref, all_ref_mask, gt, gt_mask)
        all_idx.append(idx_b)

        ttl_samples += int(gt.shape[0])
        if max_samples is not None and ttl_samples >= max_samples:
            break

    # Compute MMD
    print("[rank %s] Recon Sample size:%s Ref size: %s" % (args.rank, all_sample.size(), all_ref.size()))

    if save_dir is not None and args.save_val_results:
        smp_pcs_save_name = Path(save_dir) / f"smp_recon_pcls_gpu{args.gpu}.npy"
        ref_pcs_save_name = Path(save_dir) / f"ref_recon_pcls_gpu{args.gpu}.npy"
        smp_pcs_mask_save_name = Path(save_dir) / f"smp_recon_pcls_mask_gpu{args.gpu}.npy"
        ref_pcs_mask_save_name = Path(save_dir) / f"ref_recon_pcls_mask_gpu{args.gpu}.npy"
        np.save(smp_pcs_save_name, [s.cpu().detach().numpy() for s in all_sample])
        np.save(ref_pcs_save_name, [s.cpu().detach().numpy() for s in all_ref])
        np.save(smp_pcs_mask_save_name, [s.cpu().detach().numpy() for s in all_sample_mask])
        np.save(ref_pcs_mask_save_name, [s.cpu().detach().numpy() for s in all_ref_mask])
        print("Saving file:%s %s" % (smp_pcs_save_name, ref_pcs_save_name))

    res = emd_cd_masked(all_sample, all_sample_mask, all_ref, all_ref_mask, args.batch_size)
    cd = res['CD'] if 'CD' in res else None
    emd = res['EMD'] if 'EMD' in res else None
    print("Reconstruction CD  :%s" % cd)
    print("Reconstruction EMD :%s" % emd)
    return res


@torch.no_grad()
def validate_sample(loader, model, args, max_samples, save_dir):
    from metrics import compute_all_metrics_masked, jsd_between_point_cloud_sets as JSD
    all_sample, all_sample_mask = None, None
    all_ref, all_ref_mask = None, None
    ttl_samples = 0

    iterator = iter(loader)

    for data in iterator:
        idx_b, gt, gt_mask, gt_c = data['idx'], data['set'], data['set_mask'], data['cardinality']
        gt = gt.cuda() if args.gpu is None else gt.cuda(args.gpu)
        gt_mask = gt_mask.to(gt.device)
        gt_c = gt_c.to(gt.device)

        output = model.sample(gt_c)
        gen, gen_mask = output['set'], output['set_mask']

        # denormalize
        try:
            m, s = data['mean'].float(), data['std'].float()
            m = m.to(gt.device)
            s = s.to(gt.device)
        except (TypeError, AttributeError) as e:
            m, s = float(data['mean']), float(data['std'])

        if args.standardize_per_shape:
            offset = data['offset']
            gt = gt + offset.to(gt.device)
            gen = gen + offset.to(gen.device)

        gt = gt * s + m
        gen = gen * s + m

        all_sample, all_sample_mask = extend_batch(all_sample, all_sample_mask, gen, gen_mask)
        all_ref, all_ref_mask = extend_batch(all_ref, all_ref_mask, gt, gt_mask)

        ttl_samples += int(gt.shape[0])
        if max_samples is not None and ttl_samples >= max_samples:
            break

    print(f"[rank {args.rank}] Generation Sample size:{all_sample.size()} Ref size: {all_ref.size()}")

    if save_dir is not None and args.save_val_results:
        smp_pcs_save_name = Path(save_dir) / f"smp_syn_pcls_gpu{args.gpu}.npy"
        ref_pcs_save_name = Path(save_dir) / f"ref_syn_pcls_gpu{args.gpu}.npy"
        smp_pcs_mask_save_name = Path(save_dir) / f"smp_syn_pcls_mask_gpu{args.gpu}.npy"
        ref_pcs_mask_save_name = Path(save_dir) / f"ref_syn_pcls_mask_gpu{args.gpu}.npy"
        np.save(smp_pcs_save_name, [s.cpu().detach().numpy() for s in all_sample])
        np.save(ref_pcs_save_name, [s.cpu().detach().numpy() for s in all_ref])
        np.save(smp_pcs_mask_save_name, [s.cpu().detach().numpy() for s in all_sample_mask])
        np.save(ref_pcs_mask_save_name, [s.cpu().detach().numpy() for s in all_ref_mask])
        print(f"Saving file:{smp_pcs_save_name} {ref_pcs_save_name}")

    print(all_sample.shape, all_sample_mask.shape, all_ref.shape, all_ref_mask.shape)
    res = compute_all_metrics_masked(all_sample, all_sample_mask, all_ref, all_ref_mask, 128, accelerated_cd=True)

    if all_sample.size(-1) == 3:
        assert (~all_sample_mask).long().sum() == (~all_ref_mask).long().sum()
        all_sample = all_sample[~all_sample_mask].view(all_sample.shape[0], -1, 3)
        all_ref = all_ref[~all_ref_mask].view(all_ref.shape[0], -1, 3)
        sample_pcs = all_sample.cpu().detach().numpy()
        ref_pcs = all_ref.cpu().detach().numpy()
        jsd = JSD(sample_pcs, ref_pcs)
        jsd = torch.tensor(jsd).cuda() if args.gpu is None else torch.tensor(jsd).cuda(args.gpu)
        res.update({"JSD": jsd})
    pprint(res)
    return res


@torch.no_grad()
def visualize_reconstruct(loader, model, args, logger, epoch):
    data = next(iter(loader))
    gt, gt_mask, gt_c = data['set'], data['set_mask'], data['cardinality']
    gt = gt.cuda() if args.gpu is None else gt.cuda(args.gpu)
    gt_mask = gt_mask.to(gt.device)
    gt_c = gt_c.to(gt.device)

    # denormalize
    try:
        m, s = data['mean'].float(), data['std'].float()
        m = m[0:1, ...].to(gt.device)
        s = s[0:1, ...].to(gt.device)
    except (TypeError, AttributeError) as e:
        m, s = float(data['mean']), float(data['std'])

    output = model(gt, gt_mask)
    # model forward already do postprocessing
    recon, recon_mask = output['set'], output['set_mask']
    assert gt.shape[0] >= 16
    gt, gt_mask = pad(gt[:16, ...], gt_mask[:16, ...], model.max_outputs)

    if args.standardize_per_shape:
        offset = data['offset']
        gt = gt + offset.to(gt.device)
        recon = recon + offset.to(recon.device)

    denorm_gt = gt * s + m
    denorm_recon = recon * s + m
    logger.add_images('val_reconstruction', draw(torch.cat((denorm_gt, denorm_recon[:16, ...]), dim=0),
                                                 torch.cat((gt_mask, recon_mask[:16, ...]), dim=0)), epoch)

    # reconstruction enc/dec attention
    gt, gt_mask, gt_c = gt[:4], gt_mask[:4], gt_c[:4]
    bup = model.bottom_up(gt, gt_mask)
    tdn = model.top_down(gt_c, list(reversed(bup['features'])))
    enc_alphas, dec_alphas = bup['alphas'], tdn['alphas']
    recon, recon_mask = model.postprocess(tdn['set'], tdn['set_mask'])

    for l, a in enumerate(enc_alphas):
        alpha1, alpha2 = a
        denorm_gt = gt * s + m
        imglist1 = draw_attention(denorm_gt, gt_mask, alpha1.detach())  # bsize * Tensor([3, H, W'])
        imglist2 = draw_attention(denorm_gt, gt_mask, alpha2.detach())  # bsize * Tensor([3, H, W'])
        batch_imgs = [torch.cat([img1, img2], dim=1) for img1, img2 in zip(imglist1, imglist2)]
        for b, img in enumerate(batch_imgs):
            logger.add_image(f'val_reconstruct_att_enc{l + 1}', img,
                             epoch * len(enc_alphas) + b)  # Tensor([3, 2 * H, W'])

    for l, a in enumerate(dec_alphas):
        alpha1, alpha2 = a
        denorm_recon = recon * s + m
        imglist1 = draw_attention(denorm_recon, recon_mask, alpha1.detach())  # bsize * Tensor([3, H, W'])
        imglist2 = draw_attention(denorm_recon, recon_mask, alpha2.detach())  # bsize * Tensor([3, H, W'])
        batch_imgs = [torch.cat([img1, img2], dim=1) for img1, img2 in zip(imglist1, imglist2)]
        for b, img in enumerate(batch_imgs):
            logger.add_image(f'val_reconstruct_att_dec{l + 1}', img,
                             epoch * len(dec_alphas) + b)  # Tensor([3, 2 * H, W'])


@torch.no_grad()
def visualize_sample(loader, model, args, logger, epoch):
    data = next(iter(loader))
    gt_c = data['cardinality']
    gt_c = gt_c.cuda() if args.gpu is None else gt_c.cuda(args.gpu)

    # denormalize
    try:
        m, s = data['mean'].float(), data['std'].float()
        m = m[0:1, ...].to(gt_c.device)
        s = s[0:1, ...].to(gt_c.device)
        m_32, s_32 = m[:32, ...], s[:32, ...]
    except (TypeError, AttributeError) as e:
        m, s = float(data['mean']), float(data['std'])
        m_32, s_32 = m, s

    assert gt_c.size(0) >= 32

    output = model.sample(gt_c)
    gen, gen_mask, priors = output['set'], output['set_mask'], output['priors']
    denorm_gen = gen * s + m
    logger.add_images('val_samples', draw(denorm_gen[:32, ...], gen_mask[:32, ...]), epoch)

    if model.max_outputs == 400:
        cardinality = torch.tensor([100, 120, 180, 200, 240, 300, 350, 400] * 4).to(gt_c.device)
    elif model.max_outputs == 600:
        cardinality = torch.tensor([200, 250, 300, 350, 400, 450, 500, 550] * 4).to(gt_c.device)
    elif model.max_outputs == 30:
        cardinality = torch.tensor([3, 6, 8, 10, 13, 16, 20, 30] * 4).to(gt_c.device)
    elif model.max_outputs == 2500:
        cardinality = torch.tensor([1000, 1200, 1400, 1600, 1800, 2048, 2200, 2500] * 4).to(gt_c.device)
    else:
        cardinality = torch.tensor([1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000] * 4).to(gt_c.device)

    # inducing latents held
    z = [zml[0][0:1].repeat(cardinality.size(0), 1, 1) for zml in priors]
    output = model.sample(cardinality, given_latents=z)
    denorm_gen, gen_mask = output['set'] * s + m, output['set_mask']
    logger.add_images('val_hold_induced', draw(denorm_gen, gen_mask), epoch)

    # initial set held
    torch.manual_seed(42)
    z = [torch.stack([zml[0][0]] * 8 + [zml[0][1]] * 8 + [zml[0][2]] * 8 + [zml[0][3]] * 8, dim=0) for zml in priors]
    output = model.sample(cardinality, hold_seed=42, hold_initial_set=True, given_latents=z)
    denorm_gen, gen_mask = output['set'] * s + m, output['set_mask']
    logger.add_images('val_hold_initset', draw(denorm_gen, gen_mask), epoch)

    # draw attention
    bsize = z[0].size(0)
    for l, a in enumerate(output['alphas']):
        alpha1, alpha2 = a
        denorm_gen, gen_mask = output['set'] * s_32 + m_32, output['set_mask']
        imglist1 = draw_attention(denorm_gen, gen_mask, alpha1.detach())  # bsize * Tensor([3, H, W'])
        imglist2 = draw_attention(denorm_gen, gen_mask, alpha2.detach())  # bsize * Tensor([3, H, W'])
        batch_imgs = [torch.cat([img1, img2], dim=1) for img1, img2 in zip(imglist1, imglist2)]
        for b in range(bsize):
            logger.add_image(f'val_sample_att_dec{l + 1}', batch_imgs[b], epoch * bsize + b)  # Tensor([3, 2 * H, W'])


@torch.no_grad()
def visualize_interpolate(loader, model, args, logger, epoch):
    data = next(iter(loader))
    gt, gt_mask, gt_c = data['set'], data['set_mask'], data['cardinality']
    gt = gt.cuda() if args.gpu is None else gt.cuda(args.gpu)
    gt_mask = gt_mask.to(gt.device)
    gt_c = gt_c.to(gt.device)

    # denormalize
    try:
        m, s = data['mean'].float(), data['std'].float()
        m = m[0:1, ...].to(gt.device)
        s = s[0:1, ...].to(gt.device)
    except (TypeError, AttributeError) as e:
        m, s = float(data['mean']), float(data['std'])

    assert gt.shape[0] >= 4
    gt, gt_mask, gt_c = gt[0:4, ...], gt_mask[0:4, ...], gt_c[0:4]

    bup = model.bottom_up(gt, gt_mask)
    tdn = model.top_down(gt_c, list(reversed(bup['features'])))
    posteriors = tdn['posteriors']

    gt, gt_mask = pad(gt, gt_mask, model.max_outputs)
    cardinality = gt_c.unsqueeze(-1).repeat(1, 6)  # [4, 6]

    z = [zml[0] for zml in posteriors]  # [4, M, Dz]
    alpha = torch.arange(1, -1 / 5, -1 / 5).reshape((1, 6, 1, 1)).to(gt.device)  # [1, 6, 1, 1]
    z_ofs = [torch.cat((zl[1:, ...], zl[0:1, ...]), dim=0) for zl in z]  # [4, M, Dz]
    z_interp = [(alpha * z1.unsqueeze(1) + (1 - alpha) * z2.unsqueeze(1)).flatten(0, 1)
                for z1, z2 in zip(z, z_ofs)]  # [4, 6, M, Dz] -> [24, M, Dz]

    output = model.sample(cardinality.flatten(0, 1), given_latents=z_interp)  # [24, M, Dz]
    recon, recon_mask = output['set'], output['set_mask']
    recon_all = [gt[0:1, ...], recon[0:6, ...], gt[1:2, ...],
                 gt[1:2, ...], recon[6:12, ...], gt[2:3, ...],
                 gt[2:3, ...], recon[12:18, ...], gt[3:4, ...],
                 gt[3:4, ...], recon[18:24, ...], gt[0:1, ...]]
    recon_all_mask = [gt_mask[0:1, ...], recon_mask[0:6, ...], gt_mask[1:2, ...],
                      gt_mask[1:2, ...], recon_mask[6:12, ...], gt_mask[2:3, ...],
                      gt_mask[2:3, ...], recon_mask[12:18, ...], gt_mask[3:4, ...],
                      gt_mask[3:4, ...], recon_mask[18:24, ...], gt_mask[0:1, ...]]

    denorm_recon = torch.cat(recon_all, dim=0) * s + m
    logger.add_images('val_interpolation', draw(denorm_recon, torch.cat(recon_all_mask, dim=0)), epoch)


@torch.no_grad()
def visualize_mix(loader, model, args, logger, epoch):
    data = next(iter(loader))
    gt, gt_mask, gt_c = data['set'], data['set_mask'], data['cardinality']
    gt = gt.cuda() if args.gpu is None else gt.cuda(args.gpu)
    gt_mask = gt_mask.to(gt.device)
    gt_c = gt_c.to(gt.device)

    # denormalize
    try:
        m, s = data['mean'].float(), data['std'].float()
        m = m[0:1, ...].to(gt.device)
        s = s[0:1, ...].to(gt.device)
    except (TypeError, AttributeError) as e:
        m, s = float(data['mean']), float(data['std'])

    # 7 source B's, 2 source A's; A-mixing layer 0, 1, 2, 3
    sa, sa_mask, sa_c = gt[0:2, ...], gt_mask[0:2, ...], gt_c[0:2, ...]
    sb, sb_mask, sb_c = gt[2:9, ...], gt_mask[2:9, ...], gt_c[2:9, ...]

    bup = model.bottom_up(sa, sa_mask)
    tdn = model.top_down(sa_c, list(reversed(bup['features'])))
    za = [zml[0] for zml in tdn['posteriors']]  # [2, M, Dz]

    bup = model.bottom_up(sb, sb_mask)
    tdn = model.top_down(sb_c, list(reversed(bup['features'])))
    zb = [zml[0] for zml in tdn['posteriors']]  # [7, M, Dz]

    sa, sa_mask = pad(sa, sa_mask, model.max_outputs)
    sb, sb_mask = pad(sb, sb_mask, model.max_outputs)

    outs = list()
    out_masks = list()
    for a_idx in range(2):

        outs += [torch.ones(1, model.max_outputs, sa.size(-1)).to(sa.device) * float('inf'), sb]  # Row 1
        out_masks += [torch.ones(1, model.max_outputs).to(sa.device).bool(), sb_mask]

        for mix in range(len(tdn['posteriors'])):  # Row 2-6

            outs.append(sa[a_idx:a_idx + 1, ...])
            out_masks.append(sa_mask[a_idx:a_idx + 1, ...])

            for b_idx in range(7):
                zb_mix = [z[b_idx:b_idx + 1, ...] for z in zb]
                zb_mix[mix] = za[mix][a_idx:a_idx + 1, ...]

                output = model.sample(sb_c[b_idx:b_idx + 1], given_latents=zb_mix)
                outs.append(output['set'])
                out_masks.append(output['set_mask'])

    gen, gen_mask = torch.cat(outs, dim=0), torch.cat(out_masks, dim=0)
    denorm_gen = gen * s + m
    logger.add_images('val_mix (top-down)', draw(denorm_gen, gen_mask), epoch)


def save(model, optimizer, scheduler, epoch, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(d, path)


def resume(path, model, optimizer=None, scheduler=None, strict=True):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'], strict=strict)
    start_epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    return model, optimizer, scheduler, start_epoch
