#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List words and phrases that are among the top 1500 most frequently used words and phrases in the data, for exclusion. 

@author: Yan LIN (jackylin2012@gmail.com)
"""

import os
import gensim

KEYWORD_FOLDER = 'words' 
RESULT_PATH = 'word2vec_model'  
PILLARS = ['Climate_Change','Natural_Resources','Pollution_Waste','Ecosystem','Human_Capital','Products_and_Customers','OtherStakeholders','general']

TOP_N = 1500

pillar_keywords = {}
for pillar in PILLARS:
    with open(os.path.join(KEYWORD_FOLDER,pillar + '.txt')) as f:
        pillar_keywords[pillar] = set([x.strip() for x in f.readlines() if len(x.strip()) > 0])

esg_words = set()
for _,v in pillar_keywords.items():
    esg_words.update(v)
      
model = gensim.models.Word2Vec.load(os.path.join(RESULT_PATH,'all.word2vec'))

print(f'words in top frequent {TOP_N}')
top_set = set(list(model.wv.key_to_index.keys())[:TOP_N])
for pillar in PILLARS:
    print(f'{pillar}')
    print('*********')
    for w in pillar_keywords[pillar]:
        if w in top_set:
            print(f"{w}, count: {model.wv.get_vecattr(w,'count')}")
    print('--------')