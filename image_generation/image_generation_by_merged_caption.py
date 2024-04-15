import os

os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from diffusers import DiffusionPipeline
import numpy as np

from tqdm import tqdm
import random

import json

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    model_id = "./stable-diffusion-xl-base-1.0_fp16"
    model_id_refiner = "./stable-diffusion-xl-refiner-1.0_fp16"

    model = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True, torch_dtype=torch.float16, variant="fp16")
    model = model.to('cuda')
    refiner = DiffusionPipeline.from_pretrained(
        model_id_refiner,
        text_encoder_2=model.text_encoder_2,
        vae=model.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner = refiner.to('cuda')


    for g_idx in range(0,30): #generation for 30 times
        print('generation {} idx'.format(g_idx))
        file = "./flickr_merge/flickr30k_test_chatgpt_caps.txt"
        path = "./flickr_merge/test_fp16_refiner_{}".format(g_idx)
        setup_seed(g_idx)
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(file)
        data_list = f.readlines()

        data_save = []

        negative_prompt = "ugly, colorless , cartoon, unrealistic, low contrast,  unnatural colors, nsfw, poor proportions, additional limbs, broken fingers, broken head, broken leg, amputee, cloned faces, blurry, pixelated, double face,obscure, poor lighting, dullness, and unclear"
        with torch.no_grad():
            for idx, line in tqdm(enumerate(data_list), ncols=100, total=len(data_list)):
                one_data = json.loads(line)
                prompt = one_data['caption']
                outputs = model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, denoising_end=0.8, output_type="latent")
                image = outputs.images[0]

                image = refiner(
                    prompt=prompt,
                    num_inference_steps=50,
                    denoising_start=0.8,
                    image=image,
                ).images[0]
                image_file = os.path.join(path, one_data['image'].split('/')[-1])
                image.save(image_file)

