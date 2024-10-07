from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
import torch
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Generator
import re


def get_images_from_diffusion(
    prompt: str,
    negative_prompt: str = 'ugly, deformed, disfigured, poor details, bad anatomy',
    num_images_per_prompt: int = 100,
    seed: int = 42,
) -> Generator[Image.Image, None, str]:
    MAX_BATCH_SIZE = 10
    num_batches = num_images_per_prompt // MAX_BATCH_SIZE
    gen = torch.Generator(device='cuda').manual_seed(seed)
    for i in range(num_batches):
        batch_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=min(MAX_BATCH_SIZE, num_images_per_prompt - i * MAX_BATCH_SIZE),
            generator=gen,
        ).images
        yield from batch_images
    return 'Done'


def postprocess_object_title(object_title: str):
    """Remove special characters and make lowercase
    
    Characters not removed: `'-'` `'_'` `' '` `'.'`
    """
    object_title = object_title.replace('/', '_').replace('(', '').replace(')', '')
    object_title = object_title.replace('?', '').replace('!', '').replace(',', '')
    object_title = object_title.replace('=', '').replace('+', '').replace('#', '')
    object_title = object_title.replace('\'', '').replace('\"', '').replace('`', '')
    object_title = object_title.replace('--', '').replace('.', '').replace(':', '')
    object_title = object_title.lower().strip()
    return object_title


PROMPT_TEMPLATES = {
    'default': "{}",
    'floor': "{} laid on the floor",
    'table': "{} laid on the table",
}


def main(
    num_images_per_prompt = 100,
    prompt_type = 'default',
    seed = 42,
    rank = 0, world_size = 1,
):
    print(f'World size: {world_size}, Rank: {rank}')
    all_anns = {
        'train': json.load(open('data/vq_v2_train_anno.json')),
        'val': json.load(open('data/vq_v2_val_anno.json'))}
    object_titles = {
        split: sorted({postprocess_object_title(ann['object_title']) for ann in anns})
        for split, anns in all_anns.items()}
    prompt_template = PROMPT_TEMPLATES[prompt_type]

    p_out_root_dir = Path(f'notebooks/diffusion/')
    p_out_root_dir.mkdir(exist_ok=True, parents=True)
    for split, object_titles in object_titles.items():
        object_titles_rank = object_titles[rank::world_size]
        pbar = tqdm(object_titles_rank, desc=f'Generating images for {split} split')
        for oidx, object_title in enumerate(pbar):
            if object_title == '': continue
            if 'unsure' in object_title or 'unknown' in object_title: continue
            if not re.search(r'[a-zA-Z]', object_title): continue  # has no english characters [a-zA-Z]
            p_outdir = p_out_root_dir / prompt_type / split / object_title
            p_outdir.mkdir(exist_ok=True, parents=True)
            assert p_outdir.exists()
            tqdm.write(f'Generating images for {object_title} -> \'{p_outdir}\'')

            prompt = prompt_template.format(object_title)
            images = get_images_from_diffusion(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                seed=seed,
            )
            for i, img in enumerate(images):
                p_img = p_outdir / f'{i:05d}.png'
                img.save(p_img)


if __name__ == "__main__":
    import fire

    torch.set_float32_matmul_precision('high')
    pipe: StableDiffusionPipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        requires_safety_checker=False,
        feature_extractor=None
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.__call__

    fire.Fire(main)
