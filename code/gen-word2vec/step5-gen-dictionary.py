#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Yan LIN (jackylin2012@gmail.com)
"""

import gensim
import os
import numpy as np

RESULT_PATH = 'word2vec_model' # RESULT_PATH in the previous step
OUTPUT_PATH = '.'

TOP_N = 500
from seedwords import SEED_WORD_DICT

def sort_word_list(keywords_list:dict) -> dict:
    
    """
    Remove duplicate words belonging to two categories by assigning the word to the most relevant category.

    Parameters
    ----------
    keywords_list : dict
        DESCRIPTION.

    Returns
    -------
    keywords_list : dict
        DESCRIPTION.

    """
    word_category_simi = {} # a dict word --> list of tuples [(word, similarity),(word, similarity)]
    
    for keyword,v in keywords_list.items():
        for w in v:
            if w[0] in word_category_simi:
                word_category_simi[w[0]].append((keyword,w[1]))
            else:
                l = [(keyword,w[1])]
                word_category_simi[w[0]] = l
    
    # sort all the lists of tuples based on the similarity. The first is will have the highest.
    for k,v in word_category_simi.items():
        v.sort(reverse = True, key = lambda x:x[1])
        
    
    # refill keyword_list
    keywords_list = {}
    
    for word,category_simi in word_category_simi.items():
        category = category_simi[0][0]
        similarity = category_simi[0][1]
        if category in keywords_list:
            keywords_list[category].append((word,similarity))
        else:
            keywords_list[category]=[(word,similarity)]
    
    for k,v in keywords_list.items():
        v.sort(reverse = True, key = lambda x:x[1])
    
    results = {}
    for k,v in keywords_list.items():
        v = [x[0] for x in v]
        results[k] = v
    
    return results

def get_word_list():
    """Expanding the seed word list.
    """
    
    # Preprocessing seed words.
    seed_word_dict_new = {}
    for k,word_list in SEED_WORD_DICT.items():
        seed_word_dict_new[k] = [w.lower().strip() for w in word_list]
    seed_word_dict = seed_word_dict_new
    
    model = gensim.models.Word2Vec.load(os.path.join(RESULT_PATH,'all.word2vec'))
    
    keywords_list = {}
    for k, word_list in seed_word_dict.items():
        print(k)         
        keywords_list[k] = []
        l_simi = [] # a list of similarities, each containing the similarity of all words to the seed word. 
        for w in word_list: # iterate over all the seedword in the seedword list
            print(w)
            if w in model.wv:
                l_simi.append(1- model.wv.distances(w))
        l_simi = np.stack(l_simi,axis = 1)
        l_simi = l_simi.max(axis = 1)
        indexes = np.argpartition(l_simi,-TOP_N)[-TOP_N:]
        keywords_list[k] = list(zip([model.wv.index_to_key[x] for x in indexes],l_simi[indexes]))
        
    keywords_list = sort_word_list(keywords_list)
    for k,word_list in keywords_list.items():
        with open(os.path.join(OUTPUT_PATH,'words',k + '.txt'),'w') as f:
                f.write('\n'.join(word_list))
                
if __name__ == "__main__":
    get_word_list()
    
