import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

import torch
import numpy as np

from transformers import CLIPProcessor, CLIPModel
from transformers import AlignProcessor, AlignModel
from transformers import FlavaProcessor, FlavaModel
from tqdm import tqdm
import random
from PIL import Image
import json


from plot_tsne import plot_tsne



def evaluate_retrieval(text_feature, image_feature, gt):
    similarities = text_feature @ image_feature.t()

    topk = similarities.topk(10)[1]

    topk_correct = topk == gt

    rank1 = topk_correct[:, :1].sum() / text_image_similarities.shape[0]
    rank3 = topk_correct[:, :3].sum() / text_image_similarities.shape[0]
    rank5 = topk_correct[:, :5].sum() / text_image_similarities.shape[0]

    return rank1, rank3, rank5


def evaluate_retrieval_ndcg(text_feature, image_feature, gt):
    similarities = text_feature @ image_feature.t()
    log = torch.log2(torch.arange(2, 12).cuda())
    topk = similarities.topk(10)[1]

    topk_correct = topk == gt


    dcg_r = topk_correct / log.unsqueeze(0)
    dcg1_r = dcg_r[:,:1].sum(1)
    dcg3_r = dcg_r[:, :3].sum(1)
    dcg5_r = dcg_r[:, :5].sum(1)

    idcg = (1 / log[:2]).sum()
    ndcg1 = (dcg1_r / idcg).mean()
    ndcg3 = (dcg3_r / idcg).mean()
    ndcg5 = (dcg5_r / idcg).mean()

    return ndcg1, ndcg3, ndcg5

def get_DCG(text_feature, image_feature, image_feature_generated, gt, gt_generated):
    #print('text feature ',text_feature.shape)
    #print('image feature ',image_feature.shape)
    #print('image feature generated  ',image_feature_generated.shape)
    image_feature_ = torch.concat([image_feature, image_feature_generated])
    #print('image feature concat ', image_feature_.shape)
    similarities = text_feature @ image_feature_.t()
    #print('similarities ',similarities.shape)
    topk = similarities.topk(10)[1]
    #print('topk ',topk.shape)
    #print(topk)
    topk_correct = (topk == gt).type(torch.int)
    #print('topk_correct ',topk_correct)
    topk_correct_generated = (topk == gt_generated).type(torch.int)
    #print('topk_correct_generated ',topk_correct_generated)
    topk_correct_real_generated = topk_correct + topk_correct_generated
    #print('topk_correct_real_generated ',topk_correct_real_generated)

    log = torch.log2(torch.arange(2, 12).cuda())

    dcg = topk_correct_real_generated / log.unsqueeze(0)
    dcg5 = dcg[:,:5].sum(1)
    dcg10 = dcg[:, :10].sum(1)

    idcg = (1 / log[:2]).sum()
    ndcg5 = (dcg5 / idcg).mean()
    ndcg10 = (dcg10 / idcg).mean()

    r2 = (topk_correct_real_generated[:, :2].sum(1) == 2).sum()/text_feature.shape[0]
    r5 = (topk_correct_real_generated[:, :5].sum(1) == 2).sum()/text_feature.shape[0]
    r10 = (topk_correct_real_generated[:, :10].sum(1) == 2).sum()/text_feature.shape[0]

    ranks = similarities.sort(descending=True)[1]
    ranks_real = (ranks == torch.arange(text_image_similarities.shape[0]).unsqueeze(-1).repeat(1,ranks.shape[1]).cuda()).nonzero()[:,1]
    ranks_generated = (ranks == torch.arange(text_image_similarities.shape[0]).unsqueeze(-1).repeat(1,ranks.shape[1]).cuda() + text_image_similarities.shape[0]).nonzero()[:,1]

    mask_100 = ((ranks_real < 100) * (ranks_generated < 100))
    rank_differences = ((ranks_real - ranks_generated) * mask_100).sum()/ mask_100.sum()

    ##############compute reletavie delta###############
    r1_r = topk_correct[:, :1].sum() / text_image_similarities.shape[0]
    r3_r = topk_correct[:, :3].sum() / text_image_similarities.shape[0]
    r5_r = topk_correct[:, :5].sum() / text_image_similarities.shape[0]

    r1_g = topk_correct_generated[:, :1].sum() / text_image_similarities.shape[0]
    r3_g = topk_correct_generated[:, :3].sum() / text_image_similarities.shape[0]
    r5_g = topk_correct_generated[:, :5].sum() / text_image_similarities.shape[0]

    reletive_r1_g = 2*(r1_r.item()-r1_g.item()) / (r1_r.item()+r1_g.item())
    reletive_r3_g = 2*(r3_r.item()-r3_g.item()) / (r3_r.item()+r3_g.item())
    reletive_r5_g = 2*(r5_r.item()-r5_g.item()) / (r5_r.item()+r5_g.item())
    print(" r_delta@1: {},\n r_delta@3: {}, \n r_delta@5: {}".format(round(reletive_r1_g * 100,2), round(reletive_r3_g * 100,2), round(reletive_r5_g * 100,2)))

    log = torch.log2(torch.arange(2, 12).cuda())

    dcg_r = topk_correct / log.unsqueeze(0)
    dcg1_r = dcg_r[:,:1].sum(1)
    dcg3_r = dcg_r[:, :3].sum(1)
    dcg5_r = dcg_r[:, :5].sum(1)

    idcg = (1 / log[:2]).sum()
    ndcg1_r = (dcg1_r / idcg).mean()
    ndcg3_r = (dcg3_r / idcg).mean()
    ndcg5_r = (dcg5_r / idcg).mean()

    dcg_g = topk_correct_generated / log.unsqueeze(0)
    dcg1_g = dcg_g[:,:1].sum(1)
    dcg3_g = dcg_g[:, :3].sum(1)
    dcg5_g = dcg_g[:, :5].sum(1)

    idcg = (1 / log[:2]).sum()
    ndcg1_g = (dcg1_g / idcg).mean()
    ndcg3_g = (dcg3_g / idcg).mean()
    ndcg5_g = (dcg5_g / idcg).mean()

    reletive_n1_g = 2*(ndcg1_r.item()-ndcg1_g.item()) / (ndcg1_r.item()+ndcg1_g.item())
    reletive_n3_g = 2*(ndcg3_r.item()-ndcg3_g.item()) / (ndcg3_r.item()+ndcg3_g.item())
    reletive_n5_g = 2*(ndcg5_r.item()-ndcg5_g.item()) / (ndcg5_r.item()+ndcg5_g.item())
    print(" ndcg_delta@1: {}, \n ndcg_delta@3: {}, \n ndcg_delta@5: {}".format(round(reletive_n1_g * 100,2), round(reletive_n3_g * 100,2), round(reletive_n5_g * 100,2)))


    return r2, r5, r10, ndcg5, ndcg10, rank_differences, \
           r1_r,r3_r,r5_r,r1_g,r3_g,r5_g, \
           ndcg1_r, ndcg3_r, ndcg5_r, ndcg1_g, ndcg3_g, ndcg5_g, \
           reletive_r1_g, reletive_r3_g, reletive_r5_g, \
           reletive_n1_g, reletive_n3_g, reletive_n5_g


if __name__ == '__main__':
    image_path = "./flickr/flickr30k-images"
    image_path_generated = "./flickr_merge/test_fp16_refiner_selected_30_open_clip_h14"
    data_save_path = "./flickr_merge/selected_30_open_clip_h14_CLIP_chatgpt_selected.json"

    model = AlignModel.from_pretrained("./align-base").cuda()
    preprocesor = AlignProcessor.from_pretrained(".align-base")

    file = "./flickr/flickr30k_test.json"
    f = open(file)
    data_list = json.load(f)
    r1_sum, r3_sum, r5_sum = 0,0,0
    r1_g_sum, r3_g_sum, r5_g_sum = 0,0,0
    n1_sum, n3_sum, n5_sum = 0,0,0
    n1_g_sum, n3_g_sum, n5_g_sum = 0,0,0

    r1_sum_c, r3_sum_c, r5_sum_c = 0,0,0
    r1_g_sum_c, r3_g_sum_c, r5_g_sum_c = 0,0,0
    n1_sum_c, n3_sum_c, n5_sum_c = 0,0,0
    n1_g_sum_c, n3_g_sum_c, n5_g_sum_c = 0,0,0
    rank_differences_sum = 0
    reletive_n1_g_sum, reletive_n3_g_sum, reletive_n5_g_sum = 0 ,0 , 0
    reletive_r1_g_sum, reletive_r3_g_sum, reletive_r5_g_sum = 0, 0, 0

    for cap_idx in range(5):
        image_batch = []
        image_batch_generated = []
        text_batch = []
        id_batch = []
        batch_size = 128

        text_image_similarities = []
        selected_image = {}

        image_feature = []
        image_feature_generated = []
        text_feature = []

        model.eval()

        with torch.no_grad():
            for idx, line in tqdm(enumerate(data_list), desc="Computing Image Embeddings", ncols=100, total=len(data_list)):
                caption = line['caption'][cap_idx]

                text_batch.append(caption)
                image_file = line['image'].split('/')[-1]
                image_batch.append(preprocesor.image_processor(Image.open(os.path.join(image_path, image_file)).convert("RGB"), return_tensors='pt').data['pixel_values'][0])

                image_batch_generated.append(preprocesor.image_processor(Image.open(os.path.join(image_path_generated, image_file)).convert("RGB"), return_tensors='pt').data['pixel_values'][0])

                selected_image[image_file] = {"text": caption, "image": image_file}


                id_batch.append(image_file)

                if len(text_batch) % batch_size == 0 or idx == len(data_list) - 1 :
                    image_batch = torch.stack(image_batch).cuda()
                    image_batch_generated = torch.stack(image_batch_generated).cuda()

                    image_batch = model.get_image_features(image_batch)
                    image_batch = image_batch / image_batch.norm(p=2, dim=-1, keepdim=True)
                    #image_batch = image_batch[:, 0] #flava
                    image_batch_generated = model.get_image_features(image_batch_generated)
                    #image_batch_generated = image_batch_generated[:,0] #flava
                    image_batch_generated = image_batch_generated / image_batch_generated.norm(p=2, dim=-1, keepdim=True)

                    image_feature.append(image_batch)
                    image_feature_generated.append(image_batch_generated)

                    text_batch = preprocesor.tokenizer(text_batch, padding=True, max_length=77,truncation=True, return_tensors="pt").data
                    attn_mask = text_batch['attention_mask'].cuda()
                    text_batch = text_batch['input_ids'].cuda()
                    text_batch = model.get_text_features(text_batch, attn_mask)
                    #text_batch = text_batch[:, 0] #flava
                    text_batch = text_batch / text_batch.norm(p=2, dim=-1, keepdim=True)

                    text_feature.append(text_batch)

                    text_image_similarity = (text_batch * image_batch).sum(-1)
                    text_image_similarity_generated = (text_batch * image_batch_generated).sum(-1)
                    similarities = torch.stack([text_image_similarity, text_image_similarity_generated], dim=-1)

                    image_image_similarities = (image_batch * image_batch_generated).sum(-1)

                    text_image_similarities.append(similarities)

                    for id, similarity, image_image_similarity in zip(id_batch, similarities, image_image_similarities):
                        selected_image[id]["similarity"] = similarity.tolist()
                        selected_image[id]["image_image_similarity"] = image_image_similarity.item()

                    image_batch = []
                    image_batch_generated = []
                    text_batch = []
                    id_batch = []

        text_image_similarities = torch.concat(text_image_similarities)

        generated_max_prob = text_image_similarities.max(-1)[1].sum()/text_image_similarities.shape[0]
        print(generated_max_prob)

        text_feature = torch.cat(text_feature)
        image_feature = torch.cat(image_feature)
        image_feature_generated = torch.cat(image_feature_generated)

        feat = torch.cat([image_feature, image_feature_generated, text_feature]).cpu().numpy()
        label_test1 = [0 for index in range(1000)]
        label_test2 = [1 for index in range(1000)]
        label_test3 = [2 for index in range(1000)]

        labels = np.array(label_test1 + label_test2 + label_test3)


        #plot_tsne(feat, labels)

        gt = torch.arange(text_image_similarities.shape[0]).unsqueeze(-1).repeat(1, 10).cuda()
        gt_generated = torch.arange(text_image_similarities.shape[0]).unsqueeze(-1).repeat(1, 10).cuda() + \
                       text_image_similarities.shape[0]

        similarity_differences = text_image_similarities.mean(0)[1] - text_image_similarities.mean(0)[0]


        r1,r3,r5 = evaluate_retrieval(text_feature, image_feature, gt)
        r1_g, r3_g, r5_g = evaluate_retrieval(text_feature, image_feature_generated, gt)
        r1_sum += r1
        r3_sum += r3
        r5_sum += r5

        r1_g_sum += r1_g
        r3_g_sum += r3_g
        r5_g_sum += r5_g
        print(" r@1: {},\n r@3: {},\n r@5: {}".format(round(r1.item(),2), round(r3.item(),2), round(r5.item(),2)))
        print(" r_g@1: {},\n r_g@3: {}, \n r_g@5: {}".format(round(r1_g.item(),2), round(r3_g.item(),2), round(r5_g.item(),2)))

        n1, n3, n5 = evaluate_retrieval_ndcg(text_feature, image_feature, gt)
        n1_g, n3_g, n5_g = evaluate_retrieval_ndcg(text_feature, image_feature_generated, gt)
        n1_sum += n1
        n3_sum += n3
        n5_sum += n5

        n1_g_sum += n1_g
        n3_g_sum += n3_g
        n5_g_sum += n5_g
        print(" ndcg@1: {},\n ndcg@3: {},\n ndcg@5: {}".format(round(n1.item(), 2), round(n3.item(), 2), round(n5.item(), 2)))
        print(" ndcg_g@1: {},\n ndcg_g@3: {}, \n ndcg_g@5: {}".format(round(n1_g.item(), 2), round(n3_g.item(), 2),
                                                             round(n5_g.item(), 2)))

        r2_r_g, r5_r_g, r10_r_g, ndcg5, ndcg10, rank_differences, \
        r1_r_c, r3_r_c, r5_r_c, r1_g_c, r3_g_c, r5_g_c, \
        ndcg1_r_c, ndcg3_r_c, ndcg5_r_c, ndcg1_g_c, ndcg3_g_c, ndcg5_g_c, \
        reletive_r1_g, reletive_r3_g, reletive_r5_g, \
        reletive_n1_g, reletive_n3_g, reletive_n5_g, \
            = get_DCG(text_feature, image_feature, image_feature_generated, gt,
                      gt_generated)

        print("RankDiff: ", rank_differences.item())
        rank_differences_sum += rank_differences
        reletive_r1_g_sum += reletive_r1_g
        reletive_r3_g_sum += reletive_r3_g
        reletive_r5_g_sum += reletive_r5_g
        reletive_n1_g_sum += reletive_n1_g
        reletive_n3_g_sum += reletive_n3_g
        reletive_n5_g_sum += reletive_n5_g

        r1_sum_c += r1_r_c
        r3_sum_c += r3_r_c
        r5_sum_c += r5_r_c
        r1_g_sum_c += r1_g_c
        r3_g_sum_c += r3_g_c
        r5_g_sum_c += r5_g_c

        n1_sum_c += ndcg1_r_c
        n3_sum_c += ndcg3_r_c
        n5_sum_c += ndcg5_r_c
        n1_g_sum_c += ndcg1_g_c
        n3_g_sum_c += ndcg3_g_c
        n5_g_sum_c += ndcg5_g_c

    print('******************Five avg**************************:')
    print("RankDiff: ", rank_differences_sum / 5)
    print(" r_delta@1: {},\n r_delta@3: {}, \n r_delta@5: {}".
          format(round(reletive_r1_g_sum * 100, 2) / 5, round(reletive_r3_g_sum * 100, 2) / 5,
                 round(reletive_r5_g_sum * 100, 2) / 5))

    print(" ndcg_delta@1: {}, \n ndcg_delta@3: {}, \n ndcg_delta@5: {}".
          format(round(reletive_n1_g_sum * 100, 2) / 5, round(reletive_n3_g_sum * 100, 2) / 5,
                 round(reletive_n5_g_sum * 100, 2) / 5))

    print(" r@1: {},\n r@3: {},\n r@5: {}".format(r1_sum / 5, r3_sum / 5, r5_sum / 5))
    print(" r_g@1: {},\n r_g@3: {}, \n r_g@5: {}".format(r1_g_sum / 5, r3_g_sum / 5,
                                                         r5_g_sum / 5))
    print(" ndcg@1: {},\n ndcg@3: {},\n ndcg@5: {}".format(n1_sum / 5, n3_sum / 5, n5_sum / 5))
    print(" ndcg_g@1: {},\n ndcg_g@3: {}, \n ndcg_g@5: {}".format(n1_g_sum / 5, n3_g_sum / 5,
                                                                  n5_g_sum / 5))
    print(" r@1_c: {},\n r@3_c: {},\n r@5_c: {}".format(r1_sum_c / 5, r3_sum_c / 5, r5_sum_c / 5))
    print(" r_g@1_c: {},\n r_g@3_c: {}, \n r_g@5_c: {}".format(r1_g_sum_c / 5, r3_g_sum_c / 5,
                                                               r5_g_sum_c / 5))
    print(" ndcg@1_c: {},\n ndcg@3_c: {},\n ndcg@5_c: {}".format(n1_sum_c / 5, n3_sum_c / 5, n5_sum_c / 5))
    print(" ndcg_g@1_c: {},\n ndcg_g@3_c: {}, \n ndcg_g@5_c: {}".format(n1_g_sum_c / 5, n3_g_sum_c / 5,
                                                                        n5_g_sum_c / 5))

    f = open(data_save_path, 'w')
    f.write(json.dumps(selected_image))





