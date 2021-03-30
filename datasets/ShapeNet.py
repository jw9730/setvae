"""
Set-MultiMNIST dataset
Mostly copy-and-paste from https://github.com/stevenygd/PointFlow
"""
import os
import random
import numpy as np

import torch
import torch.utils.data

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class Uniform15KPC(torch.utils.data.Dataset):
    def __init__(self, root, subdirs, tr_sample_size=10000, te_sample_size=10000, split='train', scale=1.,
                 standardize_per_shape=False,
                 normalize_per_shape=False, random_offset=False, random_subsample=False, normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None, input_dim=3):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_offset = random_offset
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        if split == 'train':
            self.max = tr_sample_size
        elif split == 'val':
            self.max = te_sample_size
        else:
            self.max = max((tr_sample_size, te_sample_size))

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except:
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.standardize_per_shape = standardize_per_shape
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            raise NotImplementedError("normalize_per_shape==True is deprecated")
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d" % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def save_statistics(self, save_dir):
        np.save(os.path.join(save_dir, f"{self.split}_set_mean.npy"), self.all_points_mean)
        np.save(os.path.join(save_dir, f"{self.split}_set_std.npy"), self.all_points_std)
        np.save(os.path.join(save_dir, f"{self.split}_set_idx.npy"), np.array(self.shuffle_idx))

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        tr_ofs = tr_out.mean(0, keepdim=True)
        te_ofs = te_out.mean(0, keepdim=True)

        if self.standardize_per_shape:
            # If standardize_per_shape, centering in/out
            tr_out -= tr_ofs
            te_out -= te_ofs
        if self.random_offset:
            # scale data offset
            if random.uniform(0., 1.) < 0.2:
                scale = random.uniform(1., 1.5)
                tr_out -= tr_ofs
                te_out -= te_ofs
                tr_ofs *= scale
                te_ofs *= scale
                tr_out += tr_ofs
                te_out += te_ofs

        m, s = self.get_pc_stats(idx)
        m, s = torch.from_numpy(np.asarray(m)), torch.from_numpy(np.asarray(s))
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'idx': idx,
            'set': tr_out if self.split == 'train' else te_out,
            'offset': tr_ofs if self.split == 'train' else te_ofs,
            'mean': m, 'std': s, 'label': cate_idx,
            'sid': sid, 'mid': mid
        }


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root="/data/shapenet/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 standardize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_offset=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        assert 'v2' in root, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super().__init__(
            root, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            standardize_per_shape=standardize_per_shape,
            random_offset=random_offset,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
            input_dim=3)

        print(f"Done!")


def collate_fn(batch):
    ret = dict()
    for k, v in batch[0].items():
        ret.update({k: [b[k] for b in batch]})

    mean = torch.stack(ret['mean'], dim=0)  # [B, 1, 3]
    std = torch.stack(ret['std'], dim=0)  # [B, 1, 1]

    s = torch.stack(ret['set'], dim=0)  # [B, N, 3]
    offset = torch.stack(ret['offset'], dim=0)
    mask = torch.zeros(s.size(0), s.size(1)).bool()  # [B, N]
    cardinality = torch.ones(s.size(0)) * s.size(1)  # [B,]

    ret.update({'set': s, 'offset': offset, 'set_mask': mask, 'cardinality': cardinality,
                'mean': mean, 'std': std})
    return ret


def build(args):
    train_dataset = ShapeNet15kPointClouds(
        categories=args.cates,
        split='train',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        scale=args.dataset_scale,
        root=args.shapenet_data_dir,
        standardize_per_shape=args.standardize_per_shape,
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        random_subsample=True)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn,
        worker_init_fn=init_np_seed)

    val_dataset = ShapeNet15kPointClouds(
        categories=args.cates,
        split='val',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        scale=args.dataset_scale,
        root=args.shapenet_data_dir,
        standardize_per_shape=args.standardize_per_shape,
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        all_points_mean=train_dataset.all_points_mean,
        all_points_std=train_dataset.all_points_std)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False, collate_fn=collate_fn,
        worker_init_fn=init_np_seed)

    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == "__main__":
    shape_ds = ShapeNet15kPointClouds(categories=['airplane'], split='train')
    print(f"Train data statistics: mean {shape_ds.all_points_mean}, std {shape_ds.all_points_std}")
    x = shape_ds.__getitem__(0)
    for k in x.keys():
        try:
            print(f"{k}: shape {x[k].shape}")
            if x[k].shape[0] < 10:
                print(f"{k}: value {x[k]}")
        except AttributeError:
            print(f"{k}: value {x[k]}")
    x_tr, x_te = x['train_points'], x['test_points']
    print(x_tr.shape)
    print(x_te.shape)

    print(shape_ds[0])
