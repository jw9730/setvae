import cv2
import torch
import numpy as np


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
