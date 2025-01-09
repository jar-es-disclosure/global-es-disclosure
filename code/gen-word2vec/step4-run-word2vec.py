#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@author: Yan LIN (jackylin2012@gmail.com)
"""

import logging
import gensim
import pickle
import os
from gensim.models.word2vec import LineSentence

LOG_PATH = 'log'
DUMP_PATH = 'all_sentences'
NGRAM_PATH = 'bigram_trigram_model'
RESULT_PATH = 'word2vec_model'

VECTOR_SIZE = 300
WINDOW_SIZE = 5
MIN_COUNT = 5
EPOCHS = 20
THRESHOLD = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH,'run_word2vec.log')),
        logging.StreamHandler()
    ]
)

def trim_word(w,count,min_count):
    # remove single letter words and stop words
    if (w in gensim.parsing.preprocessing.STOPWORDS) or len(w) < 2:
        return gensim.utils.RULE_DISCARD
    else:
        return gensim.utils.RULE_DEFAULT

def run_word2vec():
    out_file = os.path.join(DUMP_PATH,'all.txt')
    processed_sentences = LineSentence(out_file)
    ngram_file_path = os.path.join(NGRAM_PATH,f'all_bigram_trigram_{THRESHOLD}.pkl')
    
    logging.info("Loading bigram trigram model.")
    with open(ngram_file_path,'rb') as f:
        bigram,trigram = pickle.load(f)
        
    bigram = bigram.freeze()
    trigram = trigram.freeze()

    # injecting some phrases
    bigram.phrasegrams['scope_1'] =float('inf')
    bigram.phrasegrams['scope_2'] =float('inf')
    bigram.phrasegrams['scope_3'] =float('inf')
    bigram.phrasegrams['ecological_impact'] =float('inf')
    
    bigram.phrasegrams['employee_engagement'] =float('inf')
    bigram.phrasegrams['customer_welfare'] =float('inf')
    bigram.phrasegrams['product_safety'] =float('inf')
    bigram.phrasegrams['responsible_marketing'] =float('inf')
    bigram.phrasegrams['product_quality'] =float('inf')
    
    bigram.phrasegrams['community_development'] =float('inf')
    bigram.phrasegrams['community_relation'] =float('inf')
    bigram.phrasegrams['social_capital'] =float('inf')
    bigram.phrasegrams['social_impact'] =float('inf')
    
    trigram.phrasegrams['supply_chain_sustainability']=float('inf')
    
    logging.info('Training word2vec...')
    model = gensim.models.Word2Vec(sentences=trigram[bigram[processed_sentences]],vector_size = VECTOR_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=20, epochs=EPOCHS,trim_rule = trim_word)
    model.save(os.path.join(RESULT_PATH,'all.word2vec'))
    logging.info('Model saved.')
    
if __name__ == '__main__':
    run_word2vec()
