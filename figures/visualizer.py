import torch

from draw import draw, draw_attention


def get_model_cardinality(model):
    if model.max_outputs == 400:
        cardinality = torch.tensor([100, 120, 180, 200, 240, 300, 350, 400] * 4)
    elif model.max_outputs == 600:
        cardinality = torch.tensor([200, 250, 300, 350, 400, 450, 500, 550] * 4)
    elif model.max_outputs == 2500:
        cardinality = torch.tensor([1000, 1200, 1400, 1600, 1800, 2048, 2200, 2500] * 4)
    else:
        raise NotImplementedError
    return cardinality


@torch.no_grad()
def visualize_cadinality_generalization(model, z):
    cardinality = get_model_cardinality(model)


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
        raise NotImplementedError

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