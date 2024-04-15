import os
import openai
import json
from tqdm import tqdm
import joblib
import time
openai.api_key = ""
file = "./flickr/flickr30k_test.json"
new_file = "./flickr_merge/flickr30k_test_chatgpt_caps.txt"
f = open(file)
data_list = json.load(f)

f_write = open(new_file, 'w')
data_final = []

prompt = 'Consolidate the five descriptions, avoid redundancy while including the scene described in each sentence, and make a concise summary as a prompt for text-to-image generation model to generate a image and avoid “generate” ,“summary” and "prompt":\n'
print('len is {}'.format(len(data_final)))
for idx, one_data in tqdm(enumerate(data_list), total=len(data_list)):
    print(idx)
    if idx < len(data_final):
        continue
    new_one_data = one_data.copy()
    text = ""
    for i , caption in enumerate(one_data['caption']):
        text += str(i + 1) + '.' + caption + '\n'
    text = prompt + text[:-1]
    success_flag = 0
    while success_flag == 0:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                {"role": "user", "content": text}],
                timeout=50)
            success_flag = 1

            new_text = response["choices"][0]["message"]["content"]
            new_one_data['caption'] = new_text
        except:
            print("request fail")
            success_flag = 0
    f_write.write(json.dumps(new_one_data) + "\n")
    data_final.append(new_one_data)
    joblib.dump(data_final,'./flickr_merge/flickr30k_test_chatgpt_caps')