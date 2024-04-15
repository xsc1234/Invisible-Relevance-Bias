## 使用CLIP Large 筛选生成的多个图像中最好的那个(和原图相似度最高的)

import json
import os
from shutil import copyfile
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == '__main__':
    model = CLIPModel.from_pretrained("/data/houdanyang/data/openclip_vit_h").cuda()
    preprocesor = CLIPProcessor.from_pretrained("/data/houdanyang/data/openclip_vit_h")

    file = "./flickr_merge/flickr30k_test_chatgpt_caps.txt"

    f = open(file)
    data_list = f.readlines()
    paths = []
    for i in range(0,30):
        paths.append('./flickr_merge/test_fp16_refiner_{}'.format(i))
    paths.append('./flickr/flickr30k-images')
    save_file = "./flickr_merge/selected_image_results_30_open_clip_h14.json"
    path_selected = "./flickr_merge/test_fp16_refiner_selected_30_open_clip_h14"
    if not os.path.exists(path_selected):
        os.makedirs(path_selected)

    selected_image_results = {}
    image_names = []

    caps = []
    images = []
    print('loading caption')
    for line in tqdm(data_list):
        one_data = json.loads(line)
        text = one_data['caption']
        caps.append(text)
        image_names.append(one_data['image'].split('/')[-1]) # name of the image for the caption

        for path in paths:
            image_file = os.path.join(path, one_data['image'].split('/')[-1]) #load generated image and real image
            images.append(preprocesor.image_processor(Image.open(image_file).convert("RGB"), return_tensors='pt').data['pixel_values'][0])

    images = torch.stack(images)
    images_features = []
    print('caculating similarity')
    with torch.no_grad():
        for i in tqdm(range(0, len(images), 128)):
            image_batch = images[i:i+128].cuda()
            image_batch = model.get_image_features(image_batch)
            image_batch = image_batch / image_batch.norm(p=2, dim=-1, keepdim=True)
            images_features.append(image_batch)

        images_features = torch.cat(images_features)
        images_features = images_features.reshape(len(data_list),  -1, images_features.shape[-1])
        images_features_real = images_features[:,-1,:].unsqueeze(1) #real images
        images_features_generated = images_features[:,:-1,:] #generated images
        similarities = (images_features_real * images_features_generated).sum(-1)
        _, selected_index = similarities.max(-1)

    for idx,image_name in enumerate(image_names):
        selected_image_results[image_name] = similarities[idx].tolist()
        max_similar_image_path = paths[selected_index[idx]]
        copyfile(os.path.join(max_similar_image_path, image_name), os.path.join(path_selected, image_name))

    f=open(save_file,'w')
    f.write(json.dumps(selected_image_results))
