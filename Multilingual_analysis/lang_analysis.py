import os
import pickle
import jsonlines
import json
from fastlangid.langid import LID
import fastlangid.langid as langid
import re

langid = LID()


DATA_PATH = "./Corpora/"
DATA_NAMES = [
    'openrice.txt',
    'discusshk.txt',
    'lihkg.txt',
]

def analysis_comments(data_name):
    filelist = os.listdir(DATA_PATH)
    filename = data_name

    dict_cnt = {}
    total_cnt = 0
    fp = open(DATA_PATH + filename, 'r', encoding='utf-8')
    for line in fp:
        if len(line) < 8 or '</doc>' in line or '<doc id' in line:
            continue
        total_cnt+=1
        lang = langid.predict(line)
        if dict_cnt.__contains__(lang):
            dict_cnt[lang] += 1
        else:
            dict_cnt[lang] = 1
    print(dict_cnt)
    print("total_cnt", total_cnt)


for i in range(3):
    print("start analysis: ", DATA_NAMES[i])
    analysis_comments(DATA_NAMES[i])
