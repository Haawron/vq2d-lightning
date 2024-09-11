# built-in + hydra
import time
from omegaconf import OmegaConf

# torch
import torch
import torch.utils.data

# lightning
import lightning as L

# others
import numpy as np
from tqdm import tqdm
from einops import rearrange
from transformers import Dinov2Model, BitImageProcessor

from ltvu.dataset import LitVQ2DDataModule
from ltvu.models import VQLoC

# type aliases
T_MODEL = Dinov2Model | VQLoC


def get_dino() -> Dinov2Model:
    print('Loading DINO model...')
    model: Dinov2Model = Dinov2Model.from_pretrained('facebook/dinov2-base')
    return model


def get_vqloc() -> VQLoC:
    print('Loading VQLOC model...')
    model = VQLoC(
        backbone_name='dinov2-hf',
        clip_num_frames=config.dataset.num_frames
    )
    return model


model_func = {
    'dino': get_dino,
    'vqloc': get_vqloc,
}


def forward_loops(model: T_MODEL, num_iters: int = 20, dtype = torch.float32):
    dts = []
    outputs = []
    dm = LitVQ2DDataModule(config)
    pbar = tqdm(enumerate(dm.train_dataloader(shuffle=False)), total=num_iters, leave=True)
    for batch_idx, batch in pbar:
        b, t, c, h, w = batch['segment'].shape
        if MODEL_NAME == 'vqloc':
            segments = segments.to(dtype=dtype, device='cuda')
            query = batch['query'].to(dtype=dtype, device='cuda')
            t0 = time.time()
            out = model.forward(segments, query)
        else:
            segments = rearrange(batch['segment'], 'b t c h w -> (b t) c h w')
            segments = segments.to(dtype=dtype, device='cuda')
            t0 = time.time()
            out = model.forward(segments)
        dt = time.time() - t0
        dts.append(dt)
        vram = torch.cuda.max_memory_allocated() / 1024**3  # in GB
        pbar.set_postfix({'VRAM': f'{vram:.2f} GB'})
        pbar.set_description(f'Inference time: {1000*dt:.4f} ms')
        outputs.append(out.last_hidden_state[0, 0].cpu())
        if batch_idx == num_iters:
            break
    return torch.stack(outputs), np.array(dts)


def print_time_stats(dts: np.ndarray):
    print(f'Average inference time: {1000*dts.mean():.3f} ms')
    dts = dts[dts < np.percentile(dts, 95)]
    print(f'Average inference time without outliers: {1000*dts.mean():.3f} ms')


def test_default(model: T_MODEL, num_iters: int = 20, dtype = torch.bfloat16):
    print('Testing default model')
    model = model.to(device='cuda').eval()

    outputs, dts = forward_loops(model, num_iters, dtype=dtype)
    print_time_stats(dts)
    return outputs


def test_compiled(model: T_MODEL, num_iters: int = 20, dtype = torch.bfloat16):
    print('Testing compiled model')
    # model = model.to(dtype=dtype, device='cuda').eval()
    model = model.to(device='cuda').eval()
    print('Compiling model...')
    model = torch.compile(model)
    print('Compiling done')

    outputs, dts = forward_loops(model, num_iters, dtype=dtype)
    print_time_stats(dts)
    return outputs


def main(test_idx: int = 1, num_iters: int = 20, model_name: str = 'dino'):
    L.seed_everything(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # not reproducible but faster
    torch.set_float32_matmul_precision('medium')
    torch.set_grad_enabled(False)

    assert model_name in model_func, f'Invalid model name: {model_name}'
    global MODEL_NAME
    MODEL_NAME = model_name

    model: T_MODEL = model_func[model_name]()
    if test_idx == 0:
        outputs1 = test_default(model, num_iters)
    elif test_idx == 1:
        outputs2 = test_compiled(model, num_iters)


if __name__ == '__main__':
    import fire

    MODEL_NAME = None
    config = OmegaConf.load('config/base.yaml')
    fire.Fire(main)


    # outputs1 = torch.stack(outputs1)
    # outputs2 = torch.stack(outputs2)
    # print(outputs1.norm(), outputs2.norm())
    # print((outputs1 - outputs2).abs().max())
