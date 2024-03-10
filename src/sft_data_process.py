import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import os


def sft_process():
    q_lst = []
    a_lst = []
    with open('/data/datasets/sft/GPT4-cleand/alpaca_gpt4_data_unfiltered.json', 'r', encoding='utf-8') as f:
        # data = json.load("/data/datasets/sft/GPT4-cleand/alpaca_gpt4_data_unfiltered.json")
        data = json.load(f)
        for text in tqdm(data):
            q = text['instruction']
            i = text['input']
            a = text['output']
            q = q + i
            # if len(q) < 10 or len(a) < 5:
            #     continue
            # if len(q) > 256 or len(a) > 256:
            #     continue
            q_lst.append(q)
            a_lst.append(a)

    f = open('/data/datasets/sft/belle/Belle_open_source_1M.json', 'r', encoding='utf-8')

    # s
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        q_lst.append(q)
        a_lst.append(a)
    df = pd.DataFrame(columns=['prompt', 'answer'])
    df['prompt'] = q_lst
    df['answer'] = a_lst
    df.to_csv('sft_data/sft_data.csv', index=False)
    print(df)

save_dir = './sft_data'
if not os.path.exists(save_dir): os.makedirs(save_dir)
sft_process()