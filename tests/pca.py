import sys
sys.path.append('.')

from pathlib import Path

from ltvu.lit.model import LitModule
from ltvu.lit.data import LitVQ2DDataModule

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from diffusers.utils import make_image_grid

import torch
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from imgcat import imgcat


def ten2pil(tensor, pad: float = 0.02, alpha = None, cmap = 'viridis'):
    assert tensor.dim() in (2, 3)  # (H, W), (C, H, W)
    tensor = tensor.cpu()
    tensor -= tensor.min()
    tensor /= tensor.max()

    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
        assert tensor.shape[-1] == 3

    fig = plt.figure(figsize=(5, 5))
    ax = plt.Axes(fig, [pad, pad, 1. - 2* pad, 1. - 2 * pad])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(tensor, aspect='equal', alpha=alpha, cmap=cmap)

    plots_io = BytesIO()
    fig.savefig(plots_io, format='jpg' if alpha is None else 'png', bbox_inches='tight', pad_inches=0)
    plt.close()

    img = Image.open(plots_io)
    return img


def get_models():
    path_ckpt = 'outputs/batch/2024-09-15/123741/epoch=61-prob_acc=0.7739.ckpt'
    plm_base = LitModule.load_from_checkpoint(path_ckpt).cuda()
    plm_base.eval()
    plm_base.freeze()

    path_ckpt = 'outputs/ckpts/34597/epoch=47-prob_acc=0.7977.ckpt'
    plm = LitModule.load_from_checkpoint(path_ckpt).cuda()
    plm.eval()
    plm.freeze()

    eval_config = hydra.compose(config_name='eval', overrides=[
        f'ckpt={path_ckpt.replace('=', '\\=')}',
        f'batch_size=1',
        f'num_workers=4',
        f'prefetch_factor=1'
    ])
    pdm = LitVQ2DDataModule(eval_config)  # won't use trainer here nor batched forward pass so no need to load the eval config and plm.config is enough
    pdm.batch_size = 1

    return plm_base, plm, pdm


def get_sample(pdm, bidx):
    batch = pdm.get_val_sample(idx=bidx)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    return batch


def forward(plm, pdm, batch):
    with torch.inference_mode():
        inputs = dict(**batch)
        inputs['segment'], inputs['query'] = pdm.normalize(inputs['segment'], inputs['query'])
        out = plm.model.forward(**inputs, compute_loss=True, get_intermediate_features=True, training=False)
    return out


def print_scores(out):
    # reduce_key = 'late_reduce' if 'late_reduce' in out['feat']['clip'] else 'reduce'
    # print()
    # print(out['feat']['clip']['backbone'].shape)
    # print(out['feat']['query']['backbone'].shape)
    # print(out['feat']['clip'][reduce_key].shape)
    # print(out['feat']['query'][reduce_key].shape)
    # print(out['feat']['clip']['stx'].shape)
    # if 'conv' in out['feat']['clip']:
    #     print(out['feat']['clip']['conv'].shape)

    iou = out['log_dict']['iou'].item()
    prob_acc = out['log_dict']['prob_acc'].item()

    print()
    print('=' * 80)
    print()
    print(f"mean IoU:      {iou:.3f}")
    print(f"mean prob acc: {prob_acc:.3f}")
    print()
    print('=' * 80)

    return iou, prob_acc


def feat2pcscore(feat_pca, feat_proj=None, q=4):
    *_, h, w = feat_pca.shape
    torch.manual_seed(42)
    if feat_pca.dim() == 4:
        feat_pca = rearrange(feat_pca, 't c h w -> (t h w) c')
    else:
        feat_pca = rearrange(feat_pca, 'c h w -> (h w) c')
    if feat_proj is None:
        feat_proj = feat_pca
    else:
        feat_proj = rearrange(feat_proj, 'c h w -> (h w) c')
    U, S, V = torch.pca_lowrank(feat_pca, q=q)
    _feat = feat_proj - feat_proj.mean(dim=0)
    score_map = rearrange(_feat @ V, '(h w) q -> q h w', h=h, w=w)
    return score_map


def gauss(score_map, tau=1000):
    score_map = 1. - torch.exp(-1 * score_map ** 2 / tau)
    return score_map


def gamma(score_map, gamma=1):
    score_map = score_map ** gamma
    return score_map


def remove_outlier(score_map):
    means = score_map.mean(dim=[1,2], keepdim=True)
    stds = score_map.std(dim=[1,2], keepdim=True)
    score_map = score_map.clamp(means - 2 * stds, means + 2 * stds)
    return score_map


def get_score_map_clip2query(out):
    Q = 4
    reduce_key = 'late_reduce' if 'late_reduce' in out['feat']['clip'] else 'reduce'
    print(reduce_key, 'from clip to query')
    score_map = feat2pcscore(
        out['feat']['clip'][reduce_key][..., 2:-2, 2:-2],
        out['feat']['query'][reduce_key][0][..., 2:-2, 2:-2], q=Q
    )
    score_map = remove_outlier(score_map)
    score_map = gauss(score_map)
    score_map = gamma(score_map, .9)
    images = [ten2pil(score_map[q]) for q in range(Q)]
    return images


def main(*bidxs):
    """
    Usage: 

        python tests/pca.py 0 1 2 3 4 5 6 7 8 9
    """
    GlobalHydra.instance().clear()
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("job_type", lambda : 'debug')
    OmegaConf.register_new_resolver('runtime_outdir', lambda : 'outputs/tmp')
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    hydra.initialize(config_path='../config', job_name='asdasdasdasdasd', version_base='1.3')
    p_images_save_dir = Path('outputs/pca_images/clip2query')

    plm_base, plm, pdm = get_models()

    if not bidxs:
        bidxs = [0]

    print(f'Using bidxs: {bidxs}')
    for bbb in bidxs:
        p_save_dir_b = p_images_save_dir / f'{bbb}'
        p_save_dir_b.mkdir(parents=True, exist_ok=True)
        batch = get_sample(pdm, bbb)
        fidx = len(batch['segment']) // 2
        img_seg = ten2pil(batch['segment'][0, fidx])
        img_q = ten2pil(batch['query'][0])

        img_seg.save(p_save_dir_b / f'{bbb}_seg.jpg')
        img_q.save(p_save_dir_b / f'{bbb}_q.jpg')

        print()
        imgcat(img_seg)
        print()
        imgcat(img_q)

        out_base = forward(plm_base, pdm, batch)
        out_ours = forward(plm, pdm, batch)

        iou, prob_acc = print_scores(out_base)
        iou, prob_acc = print_scores(out_ours)

        images_base = get_score_map_clip2query(out_base)
        images_ours = get_score_map_clip2query(out_ours)

        for i, img in enumerate(images_base):
            img.save(p_save_dir_b / f'{bbb}_{i:04d}_base.jpg')
        for i, img in enumerate(images_ours):
            img.save(p_save_dir_b / f'{bbb}_{i:04d}_ours.jpg')

        print()
        imgcat(make_image_grid(images_base, rows=1, cols=len(images_base), resize=256))
        print()
        imgcat(make_image_grid(images_ours, rows=1, cols=len(images_ours), resize=256))
        print()


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
