import os
from pprint import pprint
from pathlib import Path

import requests
import json

import random
import torch
import numpy as np
from tqdm import tqdm

from args import get_args
from datasets import get_datasets
from datasets.ShapeNet import collate_fn
from utils import set_random_seed
from models.networks import SetVAE
from metrics import compute_all_metrics, jsd_between_point_cloud_sets as JSD


def spreadsheet_format(result_dict):
    msg = ''
    keys = ['1-NN-CD-acc', '1-NN-EMD-acc', 'lgan_cov-CD', 'lgan_cov-EMD', 'lgan_mmd-CD', 'lgan_mmd-EMD']
    for key in keys:
        msg += f'{result_dict[key]},'
    return msg


def get_test_loader(args):
    train_dataset, val_dataset, _, _ = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        val_dataset.renormalize(mean, std)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True, drop_last=False)
    return train_loader, val_loader


def evaluate_gen(model, args):
    all_sample = list()
    all_ref = list()

    train_loader, val_loader = get_test_loader(args)
    save_dir = os.path.dirname(args.resume_checkpoint)
    save_dir = save_dir + '/train_offset' if args.eval_with_train_offset else save_dir
    iterator = iter(val_loader)
    n_batch = len(iterator)
    aux_iterator = list(iter(train_loader))[:n_batch]

    for data, data_aux in tqdm(zip(iterator, aux_iterator), total=n_batch):
        idx_b, gt, gt_mask, gt_c = data['idx'], data['set'], data['set_mask'], data['cardinality']
        gt = gt.cuda()
        gt_mask = gt_mask.cuda()
        gt_c = gt_c.cuda()

        output = model.sample(gt_c)
        gen, gen_mask = output['set'], output['set_mask']
        gen = gen[~gen_mask].reshape(gt.shape)

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda()
        s = s.cuda()

        if args.standardize_per_shape:
            if args.eval_with_train_offset:
                offset = data_aux['offset'][:len(gt)]
            else:
                offset = data['offset']
            gt = gt + offset.to(gt.device)
            gen = gen + offset.to(gen.device)

        gt = gt * s + m
        gen = gen * s + m

        all_sample.append(gen)
        all_ref.append(gt)

    all_sample = torch.cat(all_sample, 0)
    all_ref = torch.cat(all_ref, 0)

    print(f"[rank {args.rank}] Generation Sample size:{all_sample.size()} Ref size: {all_ref.size()}")

    # Save the generative output
    npy_path = Path(save_dir) / f"{args.seed}-{args.epochs - 1}"
    npy_path.mkdir(parents=True, exist_ok=True)
    smp_pcs_save_name = npy_path / f"emd_out_smp.npy"
    ref_pcs_save_name = npy_path / f"emd_out_ref.npy"
    np.save(smp_pcs_save_name, all_sample.cpu().detach().numpy())
    np.save(ref_pcs_save_name, all_ref.cpu().detach().numpy())
    print(f"Saving file: {smp_pcs_save_name} {ref_pcs_save_name}")

    print("Evaluation start")
    results = compute_all_metrics(all_sample, all_ref, 128, accelerated_cd=True)
    results = {k: (v.cpu().detach().item() if not isinstance(v, float) else v) for k, v in results.items()}
    sample_pcl_npy = all_sample.cpu().detach().numpy()
    ref_pcl_npy = all_ref.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    results.update({'JSD': jsd})

    pprint(results)
    return results


def main(args):
    model = SetVAE(args)
    model = model.cuda()
    save_dir = Path("checkpoints") / args.log_name
    args.resume_checkpoint = os.path.join(save_dir, f'checkpoint-{args.epochs - 1}.pt')
    print("Evaluate Path:{}, ".format(args.resume_checkpoint))
    checkpoint = torch.load(args.resume_checkpoint)

    model.load_state_dict(checkpoint['model'])

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.bn_mode == 'eval':
        model.eval()
    else:
        model.train()

    print(f"{args.resume_checkpoint}_{args.seed} START")
    with torch.no_grad():
        results = evaluate_gen(model, args)
    print(f"{args.resume_checkpoint}_{args.seed}" + json.dumps(results, indent=4, sort_keys=True))
    print(spreadsheet_format(results))


if __name__ == '__main__':
    args = get_args()
    # print(args)
    main(args)
