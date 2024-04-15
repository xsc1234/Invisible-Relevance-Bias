This is the code for "AI-Generated Images Introduce Invisible Relevance Bias to Text-Image Retrieval" accepted by SIGIR 2024 (Invisible Relevance Bias: Text-Image Retrieval Models Prefer AI-Generated Images)
```
@misc{xu2024aigenerated,
      title={AI-Generated Images Introduce Invisible Relevance Bias to Text-Image Retrieval}, 
      author={Shicheng Xu and Danyang Hou and Liang Pang and Jingcheng Deng and Jun Xu and Huawei Shen and Xueqi Cheng},
      year={2024},
      eprint={2311.14084},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
### Image Generation
1. Use ChatGPT to merge the captions:
```
python image_generation/call_five_captions.py
```
2. Image generation for multiple times:
```
python image_generation/image_generation_by_merged_caption.py
```
3. Selecte the most similar image:
```
python image_generation/select_most_similar_image.py
```

### Test
FLAVA, ALIGIN and BEIT-3
```
python test/comparison_{flava,aligin,beit}_5.py
```
