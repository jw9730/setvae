import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import torch
import numpy as np
import open3d as o3d

from tqdm import tqdm

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


def axis_set(ax, back_color):
    ax.set_facecolor((back_color[0], back_color[1], back_color[2], 1.))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')


def axis_set_3d(ax, back_color):
    ax.set_facecolor((back_color[0], back_color[1], back_color[2]))
    ax._axis3don = False
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.w_xaxis.set_pane_color((back_color[0], back_color[1], back_color[2], 1.))
    ax.w_yaxis.set_pane_color((back_color[0], back_color[1], back_color[2], 1.))
    ax.w_zaxis.set_pane_color((back_color[0], back_color[1], back_color[2], 1.))


def get_img(fig, H, W):
    buf = fig.canvas.buffer_rgba()
    l, b, w, h = fig.bbox.bounds
    img = np.frombuffer(buf, np.uint8).copy()
    img.shape = int(h), int(w), 4
    img = img[:, :, 0:3]
    img = cv2.resize(img, dsize=(W, H), interpolation=cv2.INTER_CUBIC)  # [H, W, 3]
    return img

def get_palette(color_opt, isize):
    if color_opt == 'set':
        palette = np.concatenate((plt.get_cmap('Set1')(range(10)),
                                  plt.get_cmap('Set2')(range(8)),
                                  plt.get_cmap('Set3')(range(12))))
        if isize > 30:
            UserWarning(f'Colormap does not support size {isize}, some elements will be aggregated')
        palette = np.concatenate((palette, palette, palette))[:isize]
    elif color_opt == 'gist_rainbow':
        palette = plt.get_cmap('gist_rainbow')(np.linspace(0., 1., isize, endpoint=False))[:, :-1]
    elif color_opt == 'pt2pc':
        palette = np.asarray([[0, 0.8, 0], [0.8, 0, 0], [0, 0.3, 0],
                              [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                              [0.3, 0.6, 0], [0.6, 0, 0.3], [0.3, 0, 0.6],
                              [0.6, 0.3, 0], [0.3, 0, 0.6], [0.6, 0, 0.3],
                              [0.8, 0.2, 0.5], [0.8, 0.2, 0.5], [0.2, 0.8, 0.5],
                              [0.2, 0.5, 0.8], [0.5, 0.2, 0.8], [0.5, 0.8, 0.2],
                              [0.3, 0.3, 0.7], [0.3, 0.7, 0.3], [0.7, 0.3, 0.3]])
        np.random.shuffle(palette)
        if isize > 21:
            UserWarning(f'Colormap does not support size {isize}, some elements will be aggregated')
        palette = np.concatenate((palette, palette, palette))[:isize]
    else:
        assert 0
        palette = np.ones([isize, 3])
        palette[:, 0] = 20.
        palette[:, 1] = 177.
        palette[:, 2] = 171.
        palette /= 255.
    return palette


@torch.no_grad()
def draw_mnist(x, x_mask, back_color=(1, 1, 1), color=(0, 0, 0), W=256):
    """ Make MNIST image
    :param x: Tensor([B, N, 2])
    :param x_mask: Tensor([B, N])
    :param back_color
    :param color
    :param W
    :return: Tensor([3 * B, H, W])
    """
    figw, figh = 16., 16.
    H = int(W * figh / figw)

    imgs = list()
    for p, m in zip(x, x_mask):
        p = p[~m, :]
        p = p.cpu()
        fig = plt.figure(figsize=(figw, figh))
        ax = fig.gca()
        axis_set(ax, back_color)

        ax.scatter(p[:, 0], -p[:, 1], color=(color[0], color[1], color[2]), s=300, ec=None)
        plt.xlim(0, 1)
        plt.ylim(-1, 0)
        fig.tight_layout()
        fig.canvas.draw()

        img = get_img(fig, H, W)
        imgs.append(torch.tensor(img).transpose(2, 0))  # [3, H, W]
        fig.canvas.flush_events()
        plt.close(fig)

    return torch.stack(imgs, dim=0)


@torch.no_grad()
def draw_attention_mnist(x, x_mask, alpha, back_color=(1, 1, 1), W=256, color_opt='gist_rainbow', dot_size=300):
    """ Compute batched images of attention weight for each attention head, with different color for each inducing point
    :param x: Tensor([B, N, 2])
    :param x_mask: Tensor([B, N])
    :param alpha: Tensor([n_heads, B, N, I])
    :param back_color
    :param W
    :return: List(Tensor([3, H, n_heads * W]))
    """
    n_heads, bsize, N, isize = alpha.shape
    figw, figh = 16., 16.
    H = int(W * figh / figw)

    palette = get_palette(color_opt, isize)

    imgs = list()
    alpha = alpha / (alpha.mean(2, keepdims=True) + 1e-3)
    for a, p, m in zip(alpha.unbind(1), x, x_mask):  # [n_heads, N, I], [N, 2], [N,]
        a = a[:, ~m, :]  # [n_heads, M, I]
        p = p[~m, :]  # [M, 2]
        a = a.cpu()
        p = p.cpu()
        for a_h in a.unbind(0):  # [M, I]
            fig = plt.figure(figsize=(figw, figh))
            ax = fig.gca()
            axis_set(ax, back_color)

            rgb = np.asarray([[palette[i, :]] for i in range(isize)])  # [I, 1, 3]
            hsv = rgb_to_hsv(rgb)
            alphas = a_h.transpose(1, 0).unsqueeze(2)  # [I, M, 1]
            alphas = alphas >= alphas.max(0)[0]
            alphas = alphas.numpy()
            hsv = (hsv * alphas).sum(axis=0)  # [M, 3]
            rgb = hsv_to_rgb(hsv)
            rgb[rgb < 0] = 0.
            rgb[rgb > 1] = 1.
            ax.scatter(p[:, 0], -p[:, 1], c=rgb, s=dot_size, ec=None)
            plt.xlim(0, 1)
            plt.ylim(-1, 0)
            fig.tight_layout()
            fig.canvas.draw()

            img = get_img(fig, H, W)
            imgs.append(torch.tensor(img).transpose(2, 0))  # [3, H, W]
            fig.canvas.flush_events()
            plt.close(fig)

    imgs = torch.stack(imgs, dim=0).reshape([bsize, n_heads, 3, H, W])  # [B, n_heads, 3, H, W]

    composed = list()
    for img_h in imgs:  # [n_heads, 3, H, W]
        composed.append(img_h)

    return composed  # bsize * Tensor([n_heads, 3, H, W])


@torch.no_grad()
def draw_pointcloud(x, x_mask, back_color=(1, 1, 1), color=(0, 0, 0), W=256):
    """ Make point cloud image
    :param x: Tensor([B, N, 3])
    :param x_mask: Tensor([B, N])
    :param back_color
    :param color
    :param W
    :return: Tensor([3 * B, W, H])
    """
    figw, figh = 16., 12.
    H = int(W * figh / figw)

    if x_mask is None:
        x_mask = [None] * x.shape[0]

    imgs = list()
    for p, m in zip(x, x_mask):
        if m is not None:
            p = p[~m, :]
        p = p.cpu()

        fig = plt.figure(figsize=(figw, figh))
        ax = fig.gca(projection='3d')
        axis_set_3d(ax, back_color)

        ax.scatter(-p[:, 2], p[:, 0], p[:, 1], color=(color[0], color[1], color[2]), marker='o', s=100)
        fig.tight_layout()
        fig.canvas.draw()

        img = get_img(fig, H, W)
        imgs.append(torch.tensor(img).transpose(2, 0).transpose(2, 1))  # [3, H, W]
        plt.close(fig)

    return torch.stack(imgs, dim=0)


@torch.no_grad()
def draw_aggregated_attention_pointcloud(x, x_mask, alpha, back_color=(1, 1, 1), W=256, mean_max=False):
    """ Compute batched images of attention weight, grid over inducing points and attention heads
    :param x: Tensor([B, N, 3])
    :param x_mask: Tensor([B, N])
    :param alpha: Tensor([n_heads, B, N, I])
    :param back_color
    :return: List(Tensor([3, W, I * H]))
    """
    figw, figh = 16., 12.
    H = int(W * figh / figw)
    n_heads, bsize, N, isize = alpha.shape

    palette = np.concatenate((plt.get_cmap('Set1')(range(10)),
                              plt.get_cmap('Set2')(range(8)),
                              plt.get_cmap('Set3')(range(12)),
                              plt.get_cmap('Set1')(range(10))))
    palette = palette[:alpha.shape[-1], :-1]
    palette = plt.get_cmap('gist_rainbow')(np.linspace(0., 1., isize, endpoint=False))[:, :-1]

    alpha = alpha / (alpha.mean(2, keepdims=True) + 1e-3)
    if mean_max:
        alpha = alpha.mean(0)
    else:
        alpha = alpha.max(dim=0)[0]
    print(alpha.shape)

    imgs = list()
    for a, p, m in zip(alpha.unbind(0), x, x_mask):  # [M, I], [N, 2], [N,]
        a = a[~m, :]  # [M, I]
        p = p[~m, :]  # [M, 2]
        a = a.cpu()
        p = p.cpu()
        fig = plt.figure(figsize=(figw, figh))
        ax = fig.gca(projection='3d')
        axis_set_3d(ax, back_color)

        rgb = np.asarray([[palette[i, :]] for i in range(isize)])  # [I, 1, 3]
        hsv = rgb_to_hsv(rgb)
        alphas = a.transpose(1, 0).unsqueeze(2)  # [I, M, 1]
        alphas = (alphas >= alphas.max(dim=0)[0]).numpy()
        hsv = (hsv * alphas).sum(axis=0)  # [M, 3]
        rgb = hsv_to_rgb(hsv)
        rgb[rgb < 0] = 0.
        rgb[rgb > 1] = 1.
        ax.scatter(-p[:, 2], p[:, 0], p[:, 1], c=rgb, marker='o', s=75)
        fig.tight_layout()
        fig.canvas.draw()

        img = get_img(fig, H, W)
        imgs.append(torch.tensor(img).transpose(2, 0).transpose(2, 1))  # [3, H, W]
        plt.close(fig)

    imgs = torch.stack(imgs, dim=0).reshape([bsize, 3, H, W])  # [B, 3, H, W]

    composed = list()
    for img_h in imgs:  # [3, H, W]
        composed.append(img_h)

    return composed  # bsize * Tensor([3, H, W])


@torch.no_grad()
def draw_attention_pointcloud(x, x_mask, alpha, back_color=(1, 1, 1), W=256, color_opt='gist_rainbow'):
    """ Compute batched images of attention weight, grid over inducing points and attention heads
    :param x: Tensor([B, N, 3])
    :param x_mask: Tensor([B, N])
    :param alpha: Tensor([n_heads, B, N, I])
    :param back_color
    :return: List(Tensor([3, n_heads * W, I * H]))
    """
    figw, figh = 16., 12.
    H = int(W * figh / figw)
    n_heads, bsize, N, isize = alpha.shape

    """
    palette = np.concatenate((plt.get_cmap('Set1')(range(10)),
                              plt.get_cmap('Set2')(range(8)),
                              plt.get_cmap('Set3')(range(12)),
                              plt.get_cmap('Set1')(range(10))))
    palette = palette[:alpha.shape[-1], :-1]
    """
    palette = plt.get_cmap('gist_rainbow')(np.linspace(0., 1., isize, endpoint=False))[:, :-1]

    alpha = alpha / (alpha.mean(2, keepdims=True) + 1e-3)
    imgs = list()
    for a, p, m in zip(alpha.unbind(1), x, x_mask):  # [n_heads, N, I], [N, 2], [N,]
        a = a[:, ~m, :]  # [n_heads, M, I]
        p = p[~m, :]  # [M, 2]
        a = a.cpu()
        p = p.cpu()
        for a_h in a.unbind(0):  # [M, I]
            fig = plt.figure(figsize=(figw, figh))
            ax = fig.gca(projection='3d')
            axis_set_3d(ax, back_color)

            rgb = np.asarray([[palette[i, :]] for i in range(isize)])  # [I, 1, 3]
            hsv = rgb_to_hsv(rgb)
            alphas = a_h.transpose(1, 0).unsqueeze(2)  # [I, M, 1]
            alphas = (alphas >= alphas.max(dim=0)[0]).numpy()
            hsv = (hsv * alphas).sum(axis=0)  # [M, 3]
            rgb = hsv_to_rgb(hsv)
            rgb[rgb < 0] = 0.
            rgb[rgb > 1] = 1.
            ax.scatter(-p[:, 2], p[:, 0], p[:, 1], c=rgb, marker='o', s=75)
            fig.tight_layout()
            fig.canvas.draw()

            img = get_img(fig, H, W)
            imgs.append(torch.tensor(img).transpose(2, 0).transpose(2, 1))  # [3, H, W]
            plt.close(fig)

    imgs = torch.stack(imgs, dim=0).reshape([bsize, n_heads, 3, H, W])  # [B, n_heads, 3, H, W]

    composed = list()
    for img_h in imgs:  # [n_heads, 3, H, W]
        composed.append(img_h)

    return composed  # bsize * Tensor([n_heads, 3, H, W])


def o3d2numpy(geo, config='camera_config.json'):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    opt = vis.get_render_option()
    opt.point_size = 10
    opt.light_on = False

    vis.add_geometry(geo)

    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('camera_config.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(True)
    vis.destroy_window()

    del vis, ctr, geo, opt

    return np.asarray(img)


@torch.no_grad()
def draw_attention_open3d(x, x_mask, alpha, config='camera_config.json', color_opt='pt2pc', view=False, size=10, palette_permutation=None):
    """ Compute batched images of attention weight, grid over inducing points and attention heads
    :param x: Tensor([B, N, 3])
    :param x_mask: Tensor([B, N])
    :param alpha: Tensor([n_heads, B, N, I])
    """
    n_heads, bsize, N, isize = alpha.shape

    palette = get_palette(color_opt, isize)
    if palette_permutation is not None:
        palette = palette[palette_permutation]

    alpha = alpha / (alpha.mean(2, keepdims=True) + 1e-3)
    imgs = list()
    for a, p, m in tqdm(zip(alpha.unbind(1), x, x_mask)):  # [n_heads, N, I], [N, 2], [N,]
        a = a[:, ~m, :]  # [n_heads, M, I]
        p = p[~m, :]  # [M, 3]
        p = p - p.mean(0, keepdim=True)
        a = a.cpu()
        p = p.cpu()
        for a_h in a.unbind(0):  # [M, I]
            # Processing points / colors
            rgb = np.asarray([[palette[i, :]] for i in range(isize)])  # [I, 1, 3]
            hsv = rgb_to_hsv(rgb)
            alphas = a_h.transpose(1, 0).unsqueeze(2)  # [I, M, 1]
            alphas = (alphas >= alphas.max(dim=0)[0]).numpy()
            hsv = (hsv * alphas).sum(axis=0)  # [M, 3]
            rgb = hsv_to_rgb(hsv)
            rgb[rgb < 0] = 0.
            rgb[rgb > 1] = 1.

            # Visualize with open3d
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(p)
            pcl.colors = o3d.utility.Vector3dVector(rgb)

            if view:
                o3d.visualization.draw_geometries([pcl])
            else:
                img = o3d2numpy(pcl, config)
                #                 plt.imshow(img)
                #                 plt.show()
                imgs.append(torch.tensor(img))
    H, W, C = imgs[0].shape
    imgs = torch.stack(imgs, dim=0).reshape([bsize, n_heads, H, W, C]).permute(0, 1, 4, 2, 3)  # [B, n_heads, 3, H, W]

    composed = list()
    for img_h in imgs:  # [n_heads, 3, H, W]
        composed.append(img_h)

    return composed  # bsize * Tensor([n_heads, 3, H, W])


@torch.no_grad()
def draw_open3d(x, x_mask, config='camera_config.json', size=0, view=False):
    """ Compute batched images of attention weight, grid over inducing points and attention heads
    :param x: Tensor([B, N, 3])
    :param x_mask: Tensor([B, N])
    :param alpha: Tensor([n_heads, B, N, I])
    """
    imgs = list()
    for p, m in tqdm(zip(x, x_mask)):  # [n_heads, N, I], [N, 2], [N,]
        p = p[~m, :]  # [M, 3]
        p = p - p.mean(0, keepdim=True)
        p = p.cpu()
        # Visualize with open3d
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(p)
        pcl.colors = o3d.utility.Vector3dVector(np.zeros_like(p))

        if view:
            o3d.visualization.draw_geometries([pcl])
        else:
            img = o3d2numpy2(pcl, config, size)
            imgs.append(torch.tensor(img))
    if view:
        return
    H, W, C = imgs[0].shape
    imgs = torch.stack(imgs, dim=0).reshape([-1, H, W, C]).permute(0, 3, 1, 2)  # [B, n_heads, 3, H, W]

    return imgs


def o3d2numpy2(geo, config='camera_config.json', size=7):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    opt = vis.get_render_option()
    opt.point_size = size

    vis.add_geometry(geo)

    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('camera_config.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(True)
    vis.destroy_window()

    del vis, ctr, geo, opt

    return np.asarray(img)


@torch.no_grad()
def draw(x, x_mask, back_color=(1, 1, 1), color=(0, 0, 0), W=256):
    if x.size(-1) == 2:
        return draw_mnist(x, x_mask, back_color, color, W)
    elif x.size(-1) == 3:
        return draw_open3d(x, x_mask, size=5)
    else:
        raise NotImplementedError


@torch.no_grad()
def draw_attention(x, x_mask, alpha, back_color=(1, 1, 1), W=256, color_opt='gist_rainbow', dot_size=None, palette_permutation=None):
    if x.size(-1) == 2:
        if dot_size is not None:
            return draw_attention_mnist(x, x_mask, alpha, back_color, W, color_opt, dot_size)
        else:
            return draw_attention_mnist(x, x_mask, alpha, back_color, W, color_opt)
    elif x.size(-1) == 3:
        return draw_attention_open3d(x, x_mask, alpha, color_opt='gist_rainbow', size=10, palette_permutation=palette_permutation)
    else:
        raise NotImplementedError
