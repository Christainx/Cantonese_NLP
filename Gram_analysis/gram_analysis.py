import os
import pickle
import jsonlines
import math
import json
from fastlangid.langid import LID
import fastlangid.langid as langid
import re

langid = LID()


def ngram_statistic():
    NGRAM = 2

    dict_Can = {}
    total_Can = 0
    set_Can = set()

    dict_tc = {}
    total_tc = 0
    set_tc = set()

    fpin = open('./Corpora/discusshk.txt', 'r', encoding = 'utf-8')
    for line in fpin:
        if len(line.strip()) > NGRAM:
            text = re.findall(r'[/u4e00-\u9fa5，。]', line)
            text = "".join(text)
            for i in range(NGRAM - 1, len(text.strip())):
                ngram = text.strip()[i - NGRAM + 1:i + 1]
                set_Can.add(ngram)
                total_Can+=1
                if dict_Can.__contains__(ngram):
                    dict_Can[ngram] += 1
                else:
                    dict_Can[ngram] = 1
    for ngram in dict_Can.keys():
        dict_Can[ngram] = dict_Can[ngram] / total_Can
    
    fpin = open('./Corpora/gigaword.txt', 'r', encoding = 'utf-8')
    for line in fpin:
        if len(line.strip()) > NGRAM:
            text = re.findall(r'[/u4e00-\u9fa5，。]', line)
            text = "".join(text)
            for i in range(NGRAM - 1, len(text.strip())):
                ngram = text.strip()[i - NGRAM + 1:i + 1]
                set_tc.add(ngram)
                total_tc+=1
                if dict_tc.__contains__(ngram):
                    dict_tc[ngram] += 1
                else:
                    dict_tc[ngram] = 1
    for ngram in dict_tc.keys():
        dict_tc[ngram] = dict_tc[ngram] / total_tc

    
    fpout = open('data_' + str(NGRAM) + 'gram.pkl', 'wb')
    pickle.dump([dict_Can, total_Can, set_Can, dict_tc, total_tc, set_tc], fpout)

    fpout = open('data_' + str(NGRAM) + 'gram.pkl', 'rb')
    [dict_Can, total_Can, set_Can, dict_tc, total_tc, set_tc] = pickle.load(fpout)

    print('len(set_Can)', len(set_Can))
    print('len(set_tc)', len(set_tc))

    fpout_Can = open('text_Can_' + str(NGRAM) + 'gram.txt', 'w', encoding='utf-8')
    fpout_tc = open('text_tc_' + str(NGRAM) + 'gram.txt', 'w', encoding='utf-8')
    interset = set_Can.intersection(set_tc)
    
    list_Can = sorted(dict_Can.items(), key=lambda d:d[1], reverse=True)
    list_tc = sorted(dict_tc.items(), key=lambda d: d[1], reverse=True)
    
    cnt = 0
    for item in list_Can:
        if item[0] in interset:
            if cnt == 1000:
                break
            cnt += 1
            item = item[0]
            fpout_Can.writelines(item + '\t' + str(dict_Can[item]) + '\t' + str(dict_tc[item]) + '\n')
    
    cnt = 0
    for item in list_tc:
        if item[0] in interset:
            if cnt == 1000:
                break
            cnt += 1
            item = item[0]
            fpout_tc.writelines(item + '\t' + str(dict_Can[item]) + '\t' + str(dict_tc[item]) + '\n')

    def entropy(list_prob):
        result = -1
        if (len(list_prob) > 0):
            result = 0;
        for x in list_prob:
            result += (-x) * math.log(x, 2)
        return result

    print(entropy(dict_Can.values()))
    print(entropy(dict_tc.values()))

ngram_statistic()